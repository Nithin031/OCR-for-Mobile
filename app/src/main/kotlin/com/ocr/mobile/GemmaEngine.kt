package com.ocr.mobile

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import com.google.mediapipe.tasks.genai.llminference.LlmInference
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.withContext
import java.io.File
import java.util.concurrent.Executors

/**
 * Gemma 3 1B IT (int4, MediaPipe .task file) running fully on the phone's CPU.
 * The model file is never downloaded by the app: the user imports it once from local storage.
 * All model calls run on one dedicated thread.
 */
class GemmaEngine(context: Context) {

    private val appContext = context.applicationContext
    private val executor = Executors.newSingleThreadExecutor()
    private val modelThread = executor.asCoroutineDispatcher()
    private var llm: LlmInference? = null

    val modelFile = File(appContext.filesDir, "models/gemma3-1b-it-int4.task")

    fun hasModel(): Boolean = modelFile.isFile && modelFile.length() > 0

    /** Copies the picked .task file into app storage. */
    suspend fun importModel(uri: Uri, onProgress: (Float) -> Unit) {
        withContext(modelThread) { unload() }
        withContext(Dispatchers.IO) {
            val resolver = appContext.contentResolver
            val total = resolver.query(uri, arrayOf(OpenableColumns.SIZE), null, null, null)?.use { c ->
                if (c.moveToFirst() && !c.isNull(0)) c.getLong(0) else -1L
            } ?: -1L
            val dir = requireNotNull(modelFile.parentFile)
            dir.mkdirs()
            val part = File(dir, modelFile.name + ".part")
            val input = resolver.openInputStream(uri) ?: error("Cannot open the selected file")
            input.use { src ->
                part.outputStream().use { dst ->
                    val buf = ByteArray(1 shl 20)
                    var copied = 0L
                    while (true) {
                        val n = src.read(buf)
                        if (n < 0) break
                        dst.write(buf, 0, n)
                        copied += n
                        if (total > 0) onProgress(copied.toFloat() / total)
                    }
                }
            }
            if (modelFile.exists()) modelFile.delete()
            check(part.renameTo(modelFile)) { "Could not save the model file" }
        }
    }

    /** Loads the model if needed. Returns the load time in ms, or 0 if it was already loaded. */
    suspend fun load(): Long = withContext(modelThread) {
        if (llm != null) return@withContext 0L
        check(hasModel()) { "No model imported yet" }
        val start = System.currentTimeMillis()
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(modelFile.absolutePath)
            .setMaxTokens(MAX_TOKENS)
            .setPreferredBackend(LlmInference.Backend.CPU)
            .build()
        llm = LlmInference.createFromOptions(appContext, options)
        System.currentTimeMillis() - start
    }

    suspend fun generate(prompt: String): String = withContext(modelThread) {
        val engine = llm ?: error("Model is not loaded")
        engine.generateResponse(prompt)
    }

    /** Frees the model on its own thread (after any running generation) without blocking the caller. */
    fun release() {
        executor.execute { unload() }
        executor.shutdown()
    }

    private fun unload() {
        llm?.close()
        llm = null
    }

    companion object {
        /** Prompt + answer tokens. 1280 is the smallest KV-cache size of the Gemma 3 1B .task variants. */
        const val MAX_TOKENS = 1280
    }
}
