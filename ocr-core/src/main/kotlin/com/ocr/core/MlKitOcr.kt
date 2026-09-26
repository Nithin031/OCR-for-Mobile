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

data class OcrLine(val text: String, val box: Rect)

data class OcrOutput(val lines: List<OcrLine>, val elapsedMs: Long)

/** On-device OCR with the bundled ML Kit Latin text recognizer (no network, no model download). */
class MlKitOcr : Closeable {

    private val recognizer = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)

    suspend fun recognize(bitmap: Bitmap): OcrOutput {
        val start = System.currentTimeMillis()
        val result = suspendCancellableCoroutine<Text> { cont ->
            recognizer.process(InputImage.fromBitmap(bitmap, 0))
                .addOnSuccessListener { text -> cont.resume(text) }
                .addOnFailureListener { e -> cont.resumeWithException(e) }
        }
        val lines = result.textBlocks
            .flatMap { block -> block.lines }
            .mapNotNull { line ->
                val box = line.boundingBox ?: return@mapNotNull null
                if (line.text.isBlank()) null else OcrLine(line.text, box)
            }
        return OcrOutput(readingOrder(lines), System.currentTimeMillis() - start)
    }

    override fun close() = recognizer.close()

    companion object {
        /**
         * Top-to-bottom, then left-to-right: lines whose vertical centres are within half the
         * median line height of the current row's centre are treated as one row.
         */
        fun readingOrder(lines: List<OcrLine>): List<OcrLine> {
            if (lines.size < 2) return lines
            val heights = lines.map { it.box.height() }.sorted()
            val tolerance = heights[heights.size / 2] * 0.5f
            val rows = mutableListOf<MutableList<OcrLine>>()
            var rowCentre = 0f
            for (line in lines.sortedBy { it.box.exactCenterY() }) {
                val cy = line.box.exactCenterY()
                if (rows.isEmpty() || cy - rowCentre > tolerance) {
                    rows.add(mutableListOf(line))
                    rowCentre = cy
                } else {
                    val row = rows.last()
                    row.add(line)
                    rowCentre = row.map { it.box.exactCenterY() }.average().toFloat()
                }
            }
            return rows.flatMap { row -> row.sortedBy { it.box.left } }
        }
    }
}
