package com.ocr.core.ppocr

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.ocr.core.OcrEngine
import com.ocr.core.OcrLine
import com.ocr.core.OcrResult
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.RotatedRect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.nio.FloatBuffer
import kotlin.math.atan2
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * PP-OCRv6 Small (det + rec ONNX, CPU) mirroring the PaddleX OCR pipeline used in Phase 0 with
 * max-side 960, no doc preprocessing and no text-line orientation. Every step names the PaddleX
 * function it mirrors. All values come from [PpOcrConfig].
 */
class PpOcrV6Engine(private val context: Context) : OcrEngine {

    override val id = ID
    override val displayName = NAME

    private var env: OrtEnvironment? = null
    private var det: OrtSession? = null
    private var rec: OrtSession? = null
    private lateinit var cfg: PpOcrConfig
    private lateinit var ctc: CtcDecoder

    override suspend fun initialize() {
        if (det != null) return
        check(OpenCVLoader.initLocal()) { "OpenCV native library failed to load" }
        val assets = context.assets
        cfg = PpOcrConfig.parse(
            assets.open("models/preprocess_config.json").bufferedReader().use { it.readText() },
            assets.open("models/pipeline_max960.json").bufferedReader().use { it.readText() },
        )
        val dict = assets.open("models/rec/ppocr_keys.txt").bufferedReader().use { r -> r.readLines() }
            .let { lines -> if (lines.lastOrNull()?.isEmpty() == true) lines.dropLast(1) else lines }
        check(dict.size == cfg.ctcClasses - 2) { "dictionary has ${dict.size} entries, expected ${cfg.ctcClasses - 2}" }
        ctc = CtcDecoder(dict, cfg.ctcBlankIndex, cfg.spaceIndex)

        val io = JSONObject(assets.open("models/model_io.json").bufferedReader().use { it.readText() })
        val environment = OrtEnvironment.getEnvironment()
        env = environment
        det = createSession(environment, "det", io)
        rec = createSession(environment, "rec", io)
    }

    private fun createSession(env: OrtEnvironment, role: String, io: JSONObject): OrtSession {
        val start = System.currentTimeMillis()
        val bytes = context.assets.open("models/$role/inference.onnx").use { it.readBytes() }
        val options = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(INTRA_OP_THREADS)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        val session = env.createSession(bytes, options)
        val spec = io.getJSONObject(role)
        val expectedIn = spec.getJSONArray("inputs").getJSONObject(0).getString("name")
        val expectedOut = spec.getJSONArray("outputs").getJSONObject(0).getString("name")
        check(session.inputNames == setOf(expectedIn)) { "$role inputs ${session.inputNames}, expected $expectedIn" }
        check(session.outputNames == setOf(expectedOut)) { "$role outputs ${session.outputNames}, expected $expectedOut" }
        if (role == "rec") {
            val shape = (session.outputInfo.getValue(expectedOut).info as TensorInfo).shape
            check(shape.last() == cfg.ctcClasses.toLong()) { "rec class dim ${shape.last()}, expected ${cfg.ctcClasses}" }
        }
        Log.d(TAG, "$role model loaded in ${System.currentTimeMillis() - start} ms (${bytes.size / 1024} KB)")
        return session
    }

    override suspend fun recognize(bitmap: Bitmap, onStage: (String) -> Unit): OcrResult {
        initialize()
        val timings = mutableListOf<Pair<String, Long>>()
        fun <T> timed(stage: String, block: () -> T): T {
            val t = System.currentTimeMillis()
            return block().also { timings += stage to System.currentTimeMillis() - t }
        }

        // Bitmap is RGBA; PaddleX decodes images with cv2 as BGR (img_mode BGR in both configs).
        val bgr = Mat()
        timed("Bitmap to BGR") {
            val rgba = Mat()
            Utils.bitmapToMat(bitmap, rgba)
            Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
            rgba.release()
        }
        try {
            onStage("Detecting text…")
            val boxes = timed("Detection (960)") { detect(bgr) }
            onStage("Recognizing ${boxes.size} lines…")
            val crops = timed("Crop ${boxes.size} lines") { boxes.map { cropMinAreaRect(bgr, it) } }
            val texts = try {
                timed("Recognition") { recognizeCrops(crops) }
            } finally {
                crops.forEach { it?.release() }
            }
            val lines = boxes.indices.mapNotNull { i ->
                val (text, score) = texts[i] ?: return@mapNotNull null
                if (score < cfg.recScoreThresh || text.isBlank()) return@mapNotNull null
                toLine(boxes[i], text, score)
            }
            return OcrResult.build(lines, timings, "PP-OCRv6 small · det max side ${cfg.detLimitSideLen} · ${boxes.size} boxes")
        } finally {
            bgr.release()
        }
    }

    // ── Detection ────────────────────────────────────────────────────────────

    /** Returns boxes as 8 floats (TL, TR, BR, BL) in original image coordinates. */
    private fun detect(bgr: Mat): List<FloatArray> {
        val srcH = bgr.rows()
        val srcW = bgr.cols()
        // DetResizeForTest.resize_image_type0 (limit_type "max").
        var ratio = if (max(srcH, srcW) > cfg.detLimitSideLen) cfg.detLimitSideLen.toDouble() / max(srcH, srcW) else 1.0
        var rh = (srcH * ratio).toInt()
        var rw = (srcW * ratio).toInt()
        if (max(rh, rw) > cfg.detMaxSideLimit) {
            ratio = cfg.detMaxSideLimit.toDouble() / max(rh, rw)
            rh = (rh * ratio).toInt()
            rw = (rw * ratio).toInt()
        }
        // Python round() is round-half-to-even, as is Math.rint.
        rh = max(Math.rint(rh / 32.0).toInt() * 32, 32)
        rw = max(Math.rint(rw / 32.0).toInt() * 32, 32)
        val resized = if (rh == srcH && rw == srcW) bgr else Mat().also {
            Imgproc.resize(bgr, it, Size(rw.toDouble(), rh.toDouble()))   // cv2 default INTER_LINEAR
        }
        Log.d(TAG, "det input shape [1, 3, $rh, $rw] from ${srcW}x$srcH")

        // NormalizeImage (order hwc): x * (scale / std[c]) - mean[c] / std[c] on BGR channel c; ToCHWImage.
        val input = FloatArray(3 * rh * rw)
        val bytes = ByteArray(rh * rw * 3)
        resized.get(0, 0, bytes)
        if (resized !== bgr) resized.release()
        val alpha = FloatArray(3) { cfg.detScale / cfg.detStd[it] }
        val beta = FloatArray(3) { -cfg.detMean[it] / cfg.detStd[it] }
        val plane = rh * rw
        for (p in 0 until plane) {
            for (c in 0 until 3) {
                input[c * plane + p] = (bytes[p * 3 + c].toInt() and 0xFF) * alpha[c] + beta[c]
            }
        }

        val pred: FloatArray
        val predH: Int
        val predW: Int
        OnnxTensor.createTensor(env, FloatBuffer.wrap(input), longArrayOf(1, 3, rh.toLong(), rw.toLong())).use { t ->
            det!!.run(mapOf(det!!.inputNames.first() to t)).use { out ->
                val tensor = out[0] as OnnxTensor
                val shape = (tensor.info as TensorInfo).shape
                predH = shape[2].toInt()
                predW = shape[3].toInt()
                pred = FloatArray(predH * predW).also { tensor.floatBuffer.get(it) }
            }
        }
        return dbPostProcess(pred, predH, predW, srcW, srcH)
    }

    /** DBPostProcess.boxes_from_bitmap (box_type quad, score_mode fast, no dilation). */
    private fun dbPostProcess(pred: FloatArray, h: Int, w: Int, destW: Int, destH: Int): List<FloatArray> {
        val mask = Mat(h, w, CvType.CV_8UC1)
        val maskBytes = ByteArray(h * w) { i -> if (pred[i] > cfg.dbThresh) 255.toByte() else 0 }
        mask.put(0, 0, maskBytes)
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        hierarchy.release()
        mask.release()

        val widthScale = destW.toDouble() / w
        val heightScale = destH.toDouble() / h
        val boxes = ArrayList<FloatArray>()
        for (index in 0 until min(contours.size, cfg.dbMaxCandidates)) {
            val contour2f = MatOfPoint2f(*contours[index].toArray())
            val rect = Imgproc.minAreaRect(contour2f)
            contour2f.release()
            val (points, sside) = miniBox(rect)
            if (sside < cfg.dbMinSize) continue
            val score = boxScoreFast(pred, h, w, points)
            if (cfg.dbBoxThresh > score) continue

            // unclip: pyclipper JT_ROUND offset of the 4-point box by d = area * ratio / perimeter, then
            // minAreaRect again. The rounded offset of a rectangle has as min-area rect the same rectangle
            // grown by d on every side, so the closed form is exact (up to pyclipper's integer rounding).
            val area = polygonArea(points)
            val length = polygonPerimeter(points)
            val d = area * cfg.dbUnclipRatio / length
            val grown = RotatedRect(rect.center, Size(rect.size.width + 2 * d, rect.size.height + 2 * d), rect.angle)
            val (box, sside2) = miniBox(grown)
            if (sside2 < cfg.dbMinSize + 2) continue

            // Map to the original image: round (half-even like numpy/Python), clip, then int16 like PaddleX.
            val mapped = FloatArray(8)
            for (k in 0 until 4) {
                mapped[2 * k] = Math.rint(box[2 * k] * widthScale).coerceIn(0.0, destW.toDouble()).toInt().toFloat()
                mapped[2 * k + 1] = Math.rint(box[2 * k + 1] * heightScale).coerceIn(0.0, destH.toDouble()).toInt().toFloat()
            }
            boxes += mapped
        }
        contours.forEach { it.release() }
        return boxes
    }

    /** DBPostProcess.get_mini_boxes: boxPoints sorted by x, then TL, TR, BR, BL; returns (points, short side). */
    private fun miniBox(rect: RotatedRect): Pair<DoubleArray, Double> {
        val pts = arrayOfNulls<Point>(4)
        rect.points(pts)
        val p = pts.map { it!! }.sortedBy { it.x }
        val (i1, i4) = if (p[1].y > p[0].y) 0 to 1 else 1 to 0
        val (i2, i3) = if (p[3].y > p[2].y) 2 to 3 else 3 to 2
        val ordered = listOf(p[i1], p[i2], p[i3], p[i4])
        val out = DoubleArray(8)
        ordered.forEachIndexed { k, pt -> out[2 * k] = pt.x; out[2 * k + 1] = pt.y }
        return out to min(rect.size.width, rect.size.height)
    }

    /** DBPostProcess.box_score_fast: mean prediction inside the (int-truncated) box polygon. */
    private fun boxScoreFast(pred: FloatArray, h: Int, w: Int, box: DoubleArray): Double {
        val xs = doubleArrayOf(box[0], box[2], box[4], box[6])
        val ys = doubleArrayOf(box[1], box[3], box[5], box[7])
        val xmin = floor(xs.min()).toInt().coerceIn(0, w - 1)
        val xmax = ceil(xs.max()).toInt().coerceIn(0, w - 1)
        val ymin = floor(ys.min()).toInt().coerceIn(0, h - 1)
        val ymax = ceil(ys.max()).toInt().coerceIn(0, h - 1)
        val mh = ymax - ymin + 1
        val mw = xmax - xmin + 1
        val mask = Mat.zeros(mh, mw, CvType.CV_8UC1)
        // numpy astype(int32) truncates toward zero.
        val poly = MatOfPoint(*Array(4) { k -> Point((xs[k] - xmin).toInt().toDouble(), (ys[k] - ymin).toInt().toDouble()) })
        Imgproc.fillPoly(mask, listOf(poly), Scalar(1.0))
        poly.release()
        val m = ByteArray(mh * mw)
        mask.get(0, 0, m)
        mask.release()
        var sum = 0.0
        var n = 0
        for (y in 0 until mh) {
            val row = (ymin + y) * w + xmin
            for (x in 0 until mw) {
                if (m[y * mw + x].toInt() != 0) {
                    sum += pred[row + x]
                    n++
                }
            }
        }
        return if (n == 0) 0.0 else sum / n
    }

    private fun polygonArea(b: DoubleArray): Double {
        var a = 0.0
        for (k in 0 until 4) {
            val j = (k + 1) % 4
            a += b[2 * k] * b[2 * j + 1] - b[2 * j] * b[2 * k + 1]
        }
        return kotlin.math.abs(a) / 2
    }

    private fun polygonPerimeter(b: DoubleArray): Double {
        var l = 0.0
        for (k in 0 until 4) {
            val j = (k + 1) % 4
            l += hypot(b[2 * j] - b[2 * k], b[2 * j + 1] - b[2 * k + 1])
        }
        return l
    }

    // ── Crop (CropByPolys, quad) ─────────────────────────────────────────────

    /** get_minarea_rect_crop + get_rotate_crop_image; null if the crop would be empty. */
    private fun cropMinAreaRect(bgr: Mat, box: FloatArray): Mat? {
        val pts = MatOfPoint2f(*Array(4) { k -> Point(box[2 * k].toInt().toDouble(), box[2 * k + 1].toInt().toDouble()) })
        val rect = Imgproc.minAreaRect(pts)
        pts.release()
        val (q, _) = miniBox(rect)
        val p = Array(4) { k -> Point(q[2 * k], q[2 * k + 1]) }
        val cropW = max(hypot(p[0].x - p[1].x, p[0].y - p[1].y), hypot(p[2].x - p[3].x, p[2].y - p[3].y)).toInt()
        val cropH = max(hypot(p[0].x - p[3].x, p[0].y - p[3].y), hypot(p[1].x - p[2].x, p[1].y - p[2].y)).toInt()
        if (cropW <= 0 || cropH <= 0) return null
        // getPerspectiveTransform takes float32 points in PaddleX.
        val src = MatOfPoint2f(*Array(4) { k -> Point(p[k].x.toFloat().toDouble(), p[k].y.toFloat().toDouble()) })
        val dst = MatOfPoint2f(
            Point(0.0, 0.0), Point(cropW.toDouble(), 0.0),
            Point(cropW.toDouble(), cropH.toDouble()), Point(0.0, cropH.toDouble()),
        )
        val m = Imgproc.getPerspectiveTransform(src, dst)
        val out = Mat()
        Imgproc.warpPerspective(
            bgr, out, m, Size(cropW.toDouble(), cropH.toDouble()),
            Imgproc.INTER_CUBIC, Core.BORDER_REPLICATE,
        )
        src.release(); dst.release(); m.release()
        if (out.rows().toDouble() / out.cols() >= 1.5) {
            val rotated = Mat()
            Core.rotate(out, rotated, Core.ROTATE_90_COUNTERCLOCKWISE)   // np.rot90
            out.release()
            return rotated
        }
        return out
    }

    // ── Recognition ──────────────────────────────────────────────────────────

    /** Crops sorted by width/height, batches of rec_batch_size, results restored to crop order. */
    private fun recognizeCrops(crops: List<Mat?>): Array<Pair<String, Float>?> {
        val results = arrayOfNulls<Pair<String, Float>>(crops.size)
        val order = crops.indices.filter { crops[it] != null }
            .sortedBy { crops[it]!!.cols() / crops[it]!!.rows().toDouble() }
        for (batch in order.chunked(cfg.recBatchSize)) {
            val inputs = batch.map { resizeNorm(crops[it]!!) }
            val maxW = inputs.maxOf { it.second }
            val h = cfg.recImageShape[1]
            // ToBatch: pad every image on the right with 0 to the batch's max width.
            val data = FloatArray(batch.size * 3 * h * maxW)
            inputs.forEachIndexed { b, (img, w) ->
                for (c in 0 until 3) for (y in 0 until h) {
                    System.arraycopy(img, (c * h + y) * w, data, ((b * 3 + c) * h + y) * maxW, w)
                }
            }
            OnnxTensor.createTensor(env, FloatBuffer.wrap(data), longArrayOf(batch.size.toLong(), 3, h.toLong(), maxW.toLong())).use { t ->
                rec!!.run(mapOf(rec!!.inputNames.first() to t)).use { out ->
                    val tensor = out[0] as OnnxTensor
                    val shape = (tensor.info as TensorInfo).shape
                    val steps = shape[1].toInt()
                    val classes = shape[2].toInt()
                    check(classes == ctc.classes) { "rec output has $classes classes, expected ${ctc.classes}" }
                    val probs = FloatArray(batch.size * steps * classes).also { tensor.floatBuffer.get(it) }
                    batch.forEachIndexed { b, idx -> results[idx] = ctc.decode(probs, b * steps * classes, steps) }
                }
            }
        }
        return results
    }

    /** OCRReisizeNormImg.resize + resize_norm_img: returns CHW floats and their width. */
    private fun resizeNorm(img: Mat): Pair<FloatArray, Int> {
        val imgH = cfg.recImageShape[1]
        val imgWMin = cfg.recImageShape[2]
        val h = img.rows()
        val w = img.cols()
        val maxWhRatio = max(imgWMin.toDouble() / imgH, w * 1.0 / h)
        var imgW = (imgH * maxWhRatio).toInt()
        val resizedW: Int
        if (imgW > cfg.recMaxWidth) {
            resizedW = cfg.recMaxWidth
            imgW = cfg.recMaxWidth
        } else {
            val ratio = w / h.toDouble()
            resizedW = if (ceil(imgH * ratio) > imgW) imgW else ceil(imgH * ratio).toInt()
        }
        val resized = Mat()
        Imgproc.resize(img, resized, Size(resizedW.toDouble(), imgH.toDouble()))   // INTER_LINEAR
        val bytes = ByteArray(imgH * resizedW * 3)
        resized.get(0, 0, bytes)
        resized.release()
        // (x / 255 - 0.5) / 0.5 in BGR channel order, CHW, zero padding on the right up to imgW.
        val out = FloatArray(3 * imgH * imgW)
        for (y in 0 until imgH) for (x in 0 until resizedW) {
            val p = (y * resizedW + x) * 3
            for (c in 0 until 3) {
                val v = (bytes[p + c].toInt() and 0xFF) * cfg.recNormScale
                out[(c * imgH + y) * imgW + x] = (v - cfg.recNormMean) / cfg.recNormStd
            }
        }
        return out to imgW
    }

    // ── Output ───────────────────────────────────────────────────────────────

    private fun toLine(q: FloatArray, text: String, score: Float): OcrLine {
        val xs = floatArrayOf(q[0], q[2], q[4], q[6])
        val ys = floatArrayOf(q[1], q[3], q[5], q[7])
        val box = Rect(xs.min().toInt(), ys.min().toInt(), xs.max().toInt(), ys.max().toInt())
        val angle = Math.toDegrees(atan2((q[3] - q[1]).toDouble(), (q[2] - q[0]).toDouble())).toFloat()
        return OcrLine(text, box, score, angle, q)
    }

    override fun close() {
        det?.close()
        rec?.close()
        det = null
        rec = null
    }

    companion object {
        const val ID = "ppocrv6"
        const val NAME = "PP-OCRv6 Small (ONNX)"
        private const val TAG = "PpOcrV6Engine"
        private const val INTRA_OP_THREADS = 4
    }
}
