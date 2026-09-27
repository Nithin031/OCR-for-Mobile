package com.ocr.core

import android.content.Context
import android.graphics.Bitmap
import com.ocr.core.ppocr.PpOcrV6Engine
import java.io.Closeable

/** An on-device OCR engine. All calls must be made off the main thread. */
interface OcrEngine : Closeable {
    val id: String
    val displayName: String

    /** Loads models; safe to call more than once. Throws if the engine cannot run. */
    suspend fun initialize() {}

    /** [onStage] receives short progress texts such as "Detecting text…". */
    suspend fun recognize(bitmap: Bitmap, onStage: (String) -> Unit = {}): OcrResult
}

/**
 * Engine output. Boxes are in the input bitmap's coordinates, lines in reading order.
 * [lowConfidence] holds indices into output.lines below [LOW_CONFIDENCE]; [timings] are (stage, ms).
 */
data class OcrResult(
    val output: OcrOutput,
    val fields: List<DocField>,
    val lowConfidence: Set<Int>,
    val timings: List<Pair<String, Long>>,
    val note: String,
) {
    val totalMs: Long get() = timings.sumOf { it.second }

    companion object {
        const val LOW_CONFIDENCE = 0.80f

        /** Orders raw lines (skew-aware), extracts fields and flags low-confidence lines. */
        fun build(rawLines: List<OcrLine>, timings: List<Pair<String, Long>>, note: String): OcrResult {
            val start = System.currentTimeMillis()
            val output = MlKitOcr.arrange(rawLines, 0)
            val fields = FieldExtractor.extract(output.rows.map { row -> row.map { it.text } })
            val low = output.lines.indices.filter { i ->
                val c = output.lines[i].confidence
                c != UNKNOWN_CONFIDENCE && c < LOW_CONFIDENCE
            }.toSet()
            val all = timings + ("Reading order + fields" to System.currentTimeMillis() - start)
            return OcrResult(output.copy(elapsedMs = all.sumOf { it.second }), fields, low, all, note)
        }
    }
}

/** The engines this build offers. ML Kit is the default and always available. */
object EngineRegistry {
    class Entry(val id: String, val displayName: String, val create: (Context) -> OcrEngine)

    const val DEFAULT_ID = MlKitEngine.ID

    val entries: List<Entry> = listOf(
        Entry(MlKitEngine.ID, MlKitEngine.NAME) { MlKitEngine() },
        Entry(MlKitEngine.HINDI_ID, MlKitEngine.HINDI_NAME) { MlKitEngine.hindi() },
        Entry(PpOcrV6Engine.ID, PpOcrV6Engine.NAME) { ctx -> PpOcrV6Engine(ctx) },
    )

    fun find(id: String): Entry = entries.firstOrNull { it.id == id } ?: entries.first { it.id == DEFAULT_ID }
}
