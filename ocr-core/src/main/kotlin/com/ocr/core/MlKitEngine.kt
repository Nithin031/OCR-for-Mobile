package com.ocr.core

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.mlkit.vision.text.TextRecognizerOptionsInterface
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlin.math.roundToInt

/**
 * ML Kit engine (bundled Latin recognizer by default, or Devanagari via [hindi]; fully offline):
 *  1. recognize the page as decoded;
 *  2. if the page is small or the first pass is weak, recognize a grayscale, contrast-stretched,
 *     upscaled copy and keep whichever pass read more text with more confidence;
 *  3. estimate page skew from line angles and order lines into rows on the de-skewed page;
 *  4. extract labelled fields and known formats (dates, IDs, phone numbers, ...).
 */
class MlKitEngine(
    override val id: String = ID,
    override val displayName: String = NAME,
    options: TextRecognizerOptionsInterface = TextRecognizerOptions.DEFAULT_OPTIONS,
) : OcrEngine {


    private val ocr = MlKitOcr(options)

    override suspend fun recognize(bitmap: Bitmap, onStage: (String) -> Unit): OcrResult {
        val timings = mutableListOf<Pair<String, Long>>()
        onStage("Reading text…")
        var t = System.currentTimeMillis()
        val first = ocr.recognizeLines(bitmap)
        timings += "ML Kit pass 1" to System.currentTimeMillis() - t
        var best = first
        var passes = 1
        var usedEnhanced = false

        val plan = ImageEnhancer.plan(bitmap)
        if (plan != null && (plan.upscale > 1f || isWeak(first))) {
            onStage("Enhancing image and reading again…")
            t = System.currentTimeMillis()
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
            timings += "Enhance + pass 2" to System.currentTimeMillis() - t
        }

        val result = OcrResult.build(best, timings, note(passes, usedEnhanced, plan))
        Log.i(TAG, "passes=$passes enhanced=$usedEnhanced plan=$plan lines=${result.output.lines.size} " +
            "skew=${result.output.skewDegrees} fields=${result.fields.size} ms=${result.totalMs}")
        return result
    }

    override fun close() = ocr.close()

    private fun note(passes: Int, usedEnhanced: Boolean, plan: ImageEnhancer.Plan?): String {
        val parts = mutableListOf(if (passes == 1) "1 pass" else "$passes passes")
        if (usedEnhanced && plan != null) {
            val how = listOfNotNull(
                "grayscale",
                if (plan.stretches) "contrast boost" else null,
                if (plan.upscale > 1f) "%.1f× upscale".format(plan.upscale) else null,
            )
            parts += "enhanced image used (${how.joinToString(", ")})"
        } else if (passes > 1) {
            parts += "original image kept"
        }
        return parts.joinToString(" · ")
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
        ),
        corners = corners?.let { c -> FloatArray(c.size) { i -> c[i] * f } },
    )

    companion object {
        const val ID = "mlkit"
        const val NAME = "ML Kit (Latin, bundled)"
        const val HINDI_ID = "mlkit_devanagari"
        const val HINDI_NAME = "ML Kit (Hindi + English, bundled)"

        /** Bundled Devanagari recognizer: reads Hindi (and Latin) text fully offline. */
        fun hindi() = MlKitEngine(HINDI_ID, HINDI_NAME, DevanagariTextRecognizerOptions.Builder().build())
        private const val TAG = "MlKitEngine"
        private const val WEAK_MEAN_CONFIDENCE = 0.75
        private const val MIN_LINES = 3
        /** The enhanced pass must be clearly better, so a tie keeps the untouched image's result. */
        private const val PREFER_ENHANCED_MARGIN = 1.05
    }
}
