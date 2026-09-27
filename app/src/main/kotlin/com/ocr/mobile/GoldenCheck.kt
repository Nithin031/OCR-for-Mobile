package com.ocr.mobile

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import com.ocr.core.OcrEngine
import org.json.JSONObject
import java.io.File
import kotlin.math.max
import kotlin.math.min

/**
 * Debug-only smoke check: runs an engine on the sample images in assets/golden/ (present only in
 * debug builds) and compares with the PaddleX max960 golden output. No scanner, no quality gate,
 * no EXIF handling: the image is decoded as-is, exactly like Phase 0.
 * Boxes are matched greedily by IoU (axis-aligned bounds) >= [MIN_IOU].
 */
object GoldenCheck {

    const val MIN_IOU = 0.5
    const val REPORT_NAME = "golden_report.txt"

    fun available(context: Context): Boolean =
        runCatching { context.assets.list("golden")?.any { it.endsWith(".json") } == true }.getOrDefault(false)

    class Box(val l: Float, val t: Float, val r: Float, val b: Float) {
        val area get() = max(0f, r - l) * max(0f, b - t)
        fun iou(o: Box): Double {
            val iw = max(0f, min(r, o.r) - max(l, o.l))
            val ih = max(0f, min(b, o.b) - max(t, o.t))
            val inter = (iw * ih).toDouble()
            val union = area + o.area - inter
            return if (union <= 0) 0.0 else inter / union
        }
    }

    class Item(val text: String, val box: Box)

    /** Greedy one-to-one matching by descending IoU. Returns pairs (goldIndex, predIndex). */
    fun match(gold: List<Item>, pred: List<Item>): List<Pair<Int, Int>> {
        val candidates = ArrayList<Triple<Double, Int, Int>>()
        for (g in gold.indices) for (p in pred.indices) {
            val iou = gold[g].box.iou(pred[p].box)
            if (iou >= MIN_IOU) candidates += Triple(iou, g, p)
        }
        candidates.sortByDescending { it.first }
        val usedG = BooleanArray(gold.size)
        val usedP = BooleanArray(pred.size)
        val out = ArrayList<Pair<Int, Int>>()
        for ((_, g, p) in candidates) if (!usedG[g] && !usedP[p]) {
            usedG[g] = true; usedP[p] = true; out += g to p
        }
        return out
    }

    fun editDistance(a: String, b: String): Int {
        var prev = IntArray(b.length + 1) { it }
        var cur = IntArray(b.length + 1)
        for (i in 1..a.length) {
            cur[0] = i
            for (j in 1..b.length) {
                val cost = if (a[i - 1] == b[j - 1]) 0 else 1
                cur[j] = minOf(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            }
            val t = prev; prev = cur; cur = t
        }
        return prev[b.length]
    }

    /** Runs the check, writes the report to app external files and returns the report text. */
    suspend fun run(context: Context, engine: OcrEngine): Pair<String, File> {
        val names = context.assets.list("golden").orEmpty().filter { it.endsWith(".json") }.sorted()
        val sb = StringBuilder()
        sb.appendLine("Golden smoke check · engine ${engine.displayName} · reference PaddleX max960")
        sb.appendLine("Match: axis-aligned IoU >= $MIN_IOU, greedy one-to-one. CER = edit distance / golden chars.")
        sb.appendLine()
        var allEdits = 0
        var allChars = 0
        for (name in names) {
            val golden = JSONObject(context.assets.open("golden/$name").bufferedReader().use { it.readText() })
            val imageName = golden.getString("image")
            val bitmap = context.assets.open("golden/$imageName").use { s ->
                BitmapFactory.decodeStream(s, null, BitmapFactory.Options().apply { inPreferredConfig = Bitmap.Config.ARGB_8888 })
            } ?: error("cannot decode $imageName")
            val result = try { engine.recognize(bitmap) } finally { bitmap.recycle() }

            val goldLines = golden.getJSONArray("lines").let { arr ->
                (0 until arr.length()).map { i ->
                    val o = arr.getJSONObject(i)
                    val pts = o.getJSONArray("box")
                    val xs = (0 until 4).map { pts.getJSONArray(it).getDouble(0).toFloat() }
                    val ys = (0 until 4).map { pts.getJSONArray(it).getDouble(1).toFloat() }
                    Item(o.getString("text"), Box(xs.min(), ys.min(), xs.max(), ys.max()))
                }
            }
            val predLines = result.output.lines.map { l ->
                val c = l.corners
                if (c != null) {
                    val xs = listOf(c[0], c[2], c[4], c[6]); val ys = listOf(c[1], c[3], c[5], c[7])
                    Item(l.text, Box(xs.min(), ys.min(), xs.max(), ys.max()))
                } else Item(l.text, Box(l.box.left.toFloat(), l.box.top.toFloat(), l.box.right.toFloat(), l.box.bottom.toFloat()))
            }
            val pairs = match(goldLines, predLines)
            val matchedG = pairs.map { it.first }.toSet()
            var edits = 0
            var chars = 0
            val diffs = StringBuilder()
            for ((g, p) in pairs.sortedBy { it.first }) {
                val gt = goldLines[g].text
                val e = editDistance(predLines[p].text, gt)
                edits += e; chars += gt.length
                if (e > 0) diffs.appendLine("    CER %.3f  gold: %s\n               ours: %s".format(e.toDouble() / max(1, gt.length), gt, predLines[p].text))
            }
            // Unmatched golden lines count as fully missed characters.
            goldLines.indices.filter { it !in matchedG }.forEach { chars += goldLines[it].text.length; edits += goldLines[it].text.length }
            allEdits += edits; allChars += chars
            val det = golden.getInt("det_box_count")
            sb.appendLine("== $imageName  (${golden.getJSONObject("image_size")})")
            sb.appendLine("  boxes: golden $det, ours ${predLines.size}, matched ${pairs.size}, " +
                "unmatched golden ${goldLines.size - pairs.size}, unmatched ours ${predLines.size - pairs.size}")
            sb.appendLine("  CER %.4f  (%d edits / %d chars)  · %d ms".format(edits.toDouble() / max(1, chars), edits, chars, result.totalMs))
            result.timings.forEach { (s, ms) -> sb.appendLine("    $s: $ms ms") }
            if (diffs.isNotEmpty()) { sb.appendLine("  lines with differences:"); sb.append(diffs) }
            sb.appendLine()
        }
        sb.appendLine("OVERALL CER %.4f  (%d edits / %d chars)".format(allEdits.toDouble() / max(1, allChars), allEdits, allChars))
        val file = File(context.getExternalFilesDir(null) ?: context.filesDir, REPORT_NAME)
        file.writeText(sb.toString())
        return sb.toString() to file
    }
}
