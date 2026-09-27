package com.ocr.core

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import kotlin.math.roundToInt

/**
 * Result of the full pipeline. Boxes in [output] are in the original bitmap's coordinates.
 * [lowConfidence] holds indices into output.lines that the recognizer was unsure about.
 */
data class PipelineResult(
    val output: OcrOutput,
    val fields: List<DocField>,
    val lowConfidence: Set<Int>,
    val passes: Int,
    val usedEnhanced: Boolean,
    val enhancePlan: ImageEnhancer.Plan?,
    val totalMs: Long,
)

/**
 * On-device OCR pipeline around ML Kit:
 *  1. recognize the page as decoded;
 *  2. if the page is small or the first pass is weak, recognize a grayscale, contrast-stretched,
 *     upscaled copy and keep whichever pass read more text with more confidence;
 *  3. estimate page skew from line angles and order lines into rows on the de-skewed page;
 *  4. extract labelled fields and known formats (dates, IDs, phone numbers, ...).
 */
class OcrPipeline(private val ocr: MlKitOcr) {

    suspend fun run(bitmap: Bitmap): PipelineResult {
        val start = System.currentTimeMillis()
        val first = ocr.recognizeLines(bitmap)
        var best = first
        var passes = 1
        var usedEnhanced = false

        val plan = ImageEnhancer.plan(bitmap)
        if (plan != null && (plan.upscale > 1f || isWeak(first))) {
            val enhanced = ImageEnhancer.apply(bitmap, plan)
            try {
                val second = ocr.recognizeLines(enhanced).map { it.scaled(1f / plan.upscale) }
                passes++
                if (score(second) > score(first) * PREFER_ENHANCED_MARGIN) {
                    best = second
                    usedEnhanced = true
                }
            } finally {
                enhanced.recycle()
            }
        }

        val elapsed = System.currentTimeMillis() - start
        val output = MlKitOcr.arrange(best, elapsed)
        val fields = FieldExtractor.extract(output.rows.map { row -> row.map { it.text } })
        val low = output.lines.indices.filter { i ->
            val c = output.lines[i].confidence
            c != UNKNOWN_CONFIDENCE && c < LOW_CONFIDENCE
        }.toSet()
        Log.i(TAG, "passes=$passes enhanced=$usedEnhanced plan=$plan lines=${output.lines.size} " +
            "skew=${output.skewDegrees} fields=${fields.size} ms=$elapsed")
        return PipelineResult(output, fields, low, passes, usedEnhanced, plan, elapsed)
    }

    private fun isWeak(lines: List<OcrLine>): Boolean {
        if (lines.size < MIN_LINES) return true
        val known = lines.filter { it.confidence != UNKNOWN_CONFIDENCE }
        return known.isNotEmpty() && known.map { it.confidence }.average() < WEAK_MEAN_CONFIDENCE
    }

    /** Letters and digits read, weighted by confidence when the recognizer reports it. */
    private fun score(lines: List<OcrLine>): Double = lines.sumOf { l ->
        val chars = l.text.count(Char::isLetterOrDigit).toDouble()
        if (l.confidence == UNKNOWN_CONFIDENCE) chars else chars * l.confidence
    }

    private fun OcrLine.scaled(f: Float): OcrLine = if (f == 1f) this else copy(
        box = Rect(
            (box.left * f).roundToInt(), (box.top * f).roundToInt(),
            (box.right * f).roundToInt(), (box.bottom * f).roundToInt(),
        )
    )

    companion object {
        private const val TAG = "OcrPipeline"
        const val LOW_CONFIDENCE = 0.5f
        private const val WEAK_MEAN_CONFIDENCE = 0.75
        private const val MIN_LINES = 3
        /** The enhanced pass must be clearly better, so a tie keeps the untouched image's result. */
        private const val PREFER_ENHANCED_MARGIN = 1.05
    }
}
