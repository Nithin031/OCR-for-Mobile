package com.ocr.core

import android.graphics.Bitmap
import android.graphics.Rect
import android.util.Log
import com.google.mlkit.vision.text.devanagari.DevanagariTextRecognizerOptions
import kotlin.math.roundToInt

/**
 * English + Hindi, automatically: the full ML Kit Latin pipeline ([MlKitEngine]) plus the bundled
 * Devanagari recognizer on the same page, merged with [ScriptMerge]. Lines with real Hindi text come from
 * the Devanagari pass; everything else is the Latin result unchanged. Fully offline.
 */
class MlKitAutoEngine : OcrEngine {

    override val id = ID
    override val displayName = NAME

    private val latin = MlKitEngine()
    private val devanagari = MlKitOcr(DevanagariTextRecognizerOptions.Builder().build())

    override suspend fun recognize(bitmap: Bitmap, onStage: (String) -> Unit): OcrResult {
        val base = latin.recognize(bitmap, onStage)

        onStage("Reading Hindi text…")
        val start = System.currentTimeMillis()
        // Small pages read better upscaled, as in the Latin pipeline; boxes are mapped back to the page.
        val plan = ImageEnhancer.plan(bitmap)?.takeIf { it.upscale > 1f }
        val hindiLines = if (plan == null) {
            devanagari.recognizeLines(bitmap)
        } else {
            val enhanced = ImageEnhancer.apply(bitmap, plan)
            try {
                devanagari.recognizeLines(enhanced).map { it.scaled(1f / plan.upscale) }
            } finally {
                enhanced.recycle()
            }
        }
        val ms = System.currentTimeMillis() - start

        val latinLines = base.output.lines
        val (keepLatin, useHindi) = ScriptMerge.merge(
            latinLines.map { it.text to it.box.toBox() },
            hindiLines.map { it.text to it.box.toBox() },
        )
        if (useHindi.isEmpty()) {
            // No Hindi on the page: the Latin result as it was, plus the time the check took.
            return base.copy(timings = base.timings + ("Hindi check" to ms))
        }
        val merged = keepLatin.map { latinLines[it] } + useHindi.map { hindiLines[it] }
        Log.i(TAG, "Hindi lines ${useHindi.size}, Latin lines replaced ${latinLines.size - keepLatin.size}")
        val timings = base.timings.filterNot { it.first == "Reading order + fields" } + ("Hindi pass (Devanagari)" to ms)
        return OcrResult.build(merged, timings, "${base.note} · ${useHindi.size} lines read as Hindi")
    }

    override fun close() {
        latin.close()
        devanagari.close()
    }

    private fun Rect.toBox() = ScriptMerge.Box(left.toFloat(), top.toFloat(), right.toFloat(), bottom.toFloat())

    private fun OcrLine.scaled(f: Float): OcrLine = copy(
        box = Rect((box.left * f).roundToInt(), (box.top * f).roundToInt(), (box.right * f).roundToInt(), (box.bottom * f).roundToInt()),
        corners = corners?.let { c -> FloatArray(c.size) { i -> c[i] * f } },
    )

    companion object {
        const val ID = "mlkit_auto"
        const val NAME = "ML Kit (English + Hindi, auto)"
        private const val TAG = "MlKitAutoEngine"
    }
}
