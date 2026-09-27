package com.ocr.core

import android.graphics.Bitmap
import android.graphics.Rect
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.suspendCancellableCoroutine
import java.io.Closeable
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException

/**
 * One recognized line. [confidence] is ML Kit's 0..1 score, or [UNKNOWN_CONFIDENCE] when the model gives
 * none; [angle] is the line's rotation in degrees (clockwise positive).
 */
data class OcrLine(
    val text: String,
    val box: Rect,
    val confidence: Float = UNKNOWN_CONFIDENCE,
    val angle: Float = 0f,
)

const val UNKNOWN_CONFIDENCE = -1f

/** Lines in reading order; [rows] groups the same lines into visual rows. [skewDegrees] is the page tilt used. */
data class OcrOutput(
    val lines: List<OcrLine>,
    val elapsedMs: Long,
    val rows: List<List<OcrLine>> = lines.map { listOf(it) },
    val skewDegrees: Float = 0f,
)

/** On-device OCR with the bundled ML Kit Latin text recognizer (no network, no model download). */
class MlKitOcr : Closeable {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    suspend fun recognize(bitmap: Bitmap): OcrOutput {
        val start = System.currentTimeMillis()
        val lines = recognizeLines(bitmap)
        return arrange(lines, System.currentTimeMillis() - start)
    }

    /** Raw lines (unordered) in the bitmap's coordinates; blank lines and lines without a box are dropped. */
    suspend fun recognizeLines(bitmap: Bitmap): List<OcrLine> {
        val result = suspendCancellableCoroutine<Text> { cont ->
            recognizer.process(InputImage.fromBitmap(bitmap, 0))
                .addOnSuccessListener { text -> cont.resume(text) }
                .addOnFailureListener { e -> cont.resumeWithException(e) }
        }
        return result.textBlocks
            .flatMap { block -> block.lines }
            .mapNotNull { line ->
                val box = line.boundingBox ?: return@mapNotNull null
                if (line.text.isBlank()) return@mapNotNull null
                val conf = line.confidence
                OcrLine(line.text, box, if (conf > 0f) conf else UNKNOWN_CONFIDENCE, line.angle)
            }
    }

    override fun close() = recognizer.close()

    companion object {
        /** Estimates page skew from the line angles and puts the lines in reading order. */
        fun arrange(lines: List<OcrLine>, elapsedMs: Long): OcrOutput {
            val skew = ReadingOrder.estimateSkew(lines.map { it.angle })
            val rows = readingRows(lines, skew)
            return OcrOutput(rows.flatten(), elapsedMs, rows, skew)
        }

        /** Top-to-bottom, then left-to-right (see [ReadingOrder.rows]); no skew correction. */
        fun readingOrder(lines: List<OcrLine>): List<OcrLine> = readingRows(lines, 0f).flatten()

        private fun readingRows(lines: List<OcrLine>, skewDeg: Float): List<List<OcrLine>> =
            ReadingOrder.rows(lines, skewDeg) { l ->
                ReadingOrder.Geo(
                    left = l.box.left.toFloat(),
                    centreX = l.box.exactCenterX(),
                    centreY = l.box.exactCenterY(),
                    height = l.box.height().toFloat(),
                )
            }
    }
}
