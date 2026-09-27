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
 * On-device Gemma (MediaPipe LLM Inference, CPU): Gemma 3 1B IT int4 (.task) or Gemma 4 E2B (.litertlm).
 * Model files are never downloaded by the app: the user imports them once from local storage. Each
 * imported file keeps its own name under filesDir/models; one of them is active.
 * All model calls run on one dedicated thread.
 */
class GemmaEngine(context: Context) {

    private val appContext = context.applicationContext
    private val executor = Executors.newSingleThreadExecutor()
    private val modelThread = executor.asCoroutineDispatcher()
    private val prefs = appContext.getSharedPreferences("gemma", Context.MODE_PRIVATE)
    private var llm: LlmInference? = null
    private var loadedFile: File? = null

    private val modelDir = File(appContext.filesDir, "models")

    /** Imported model files, newest first. */
    fun models(): List<File> =
        modelDir.listFiles { f -> f.isFile && f.length() > 0 && f.extension.lowercase() in EXTENSIONS }
            ?.sortedByDescending { it.lastModified() }.orEmpty()

    /** The active model: the one chosen last, else the newest imported. */
    val modelFile: File?
        get() {
            val all = models()
            return all.firstOrNull { it.name == prefs.getString(KEY_ACTIVE, null) }
                ?: all.firstOrNull { !hasFailed(it.name) } ?: all.firstOrNull()
        }

    fun hasModel(): Boolean = modelFile != null

    /** Choosing a model explicitly also gives a previously crashed model another try. */
    fun setActive(name: String) {
        val failed = prefs.getStringSet(KEY_FAILED, emptySet()).orEmpty() - name
        prefs.edit().putString(KEY_ACTIVE, name).putStringSet(KEY_FAILED, failed).apply()
    }

    /** Copies the picked .task/.litertlm file into app storage under its own name and makes it active. */
    suspend fun importModel(uri: Uri, onProgress: (Float) -> Unit) {
        withContext(modelThread) { unload() }
        val name = withContext(Dispatchers.IO) {
            val resolver = appContext.contentResolver
            var total = -1L
            var displayName: String? = null
            resolver.query(uri, arrayOf(OpenableColumns.SIZE, OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
                if (c.moveToFirst()) {
                    if (!c.isNull(0)) total = c.getLong(0)
                    if (!c.isNull(1)) displayName = c.getString(1)
                }
            }
            val fileName = safeFileName(displayName ?: "")
            val target = File(modelDir, fileName)
            modelDir.mkdirs()
            val part = File(modelDir, "$fileName.part")
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
            if (target.exists()) target.delete()
            check(part.renameTo(target)) { "Could not save the model file" }
            fileName
        }
        setActive(name)
    }

    /** Loads the active model if needed. Returns the load time in ms, or 0 if it was already loaded. */
    suspend fun load(): Long = withContext(modelThread) {
        val file = modelFile ?: error("No model imported yet")
        if (llm != null && loadedFile == file) return@withContext 0L
        unload()
        val start = System.currentTimeMillis()
        val options = LlmInference.LlmInferenceOptions.builder()
            .setModelPath(file.absolutePath)
            .setMaxTokens(MAX_TOKENS)
            .setPreferredBackend(LlmInference.Backend.CPU)
            .build()
        // A native crash in the engine kills the process and cannot be caught here, so leave a marker
        // that the next start can see (see recoverFromCrash()).
        prefs.edit().putString(KEY_IN_FLIGHT, file.name).commit()
        llm = try {
            LlmInference.createFromOptions(appContext, options)
        } catch (e: Throwable) {
            prefs.edit().remove(KEY_IN_FLIGHT).apply()   // an ordinary error, not a crash
            throw e
        }
        loadedFile = file
        System.currentTimeMillis() - start
    }

    suspend fun generate(prompt: String): String = withContext(modelThread) {
        val engine = llm ?: error("Model is not loaded")
        loadedFile?.let { prefs.edit().putString(KEY_IN_FLIGHT, it.name).commit() }
        try {
            engine.generateResponse(prompt)
        } finally {
            prefs.edit().remove(KEY_IN_FLIGHT).apply()   // reached unless the process died
        }
    }

    /**
     * If the app died while a model was loading or answering, that model is set aside and another
     * imported model becomes active. Returns the crashed model's file name, or null.
     */
    fun recoverFromCrash(): String? {
        val crashed = prefs.getString(KEY_IN_FLIGHT, null) ?: return null
        val failed = prefs.getStringSet(KEY_FAILED, emptySet()).orEmpty() + crashed
        val fallback = models().firstOrNull { it.name !in failed }?.name
        prefs.edit().remove(KEY_IN_FLIGHT).putStringSet(KEY_FAILED, failed)
            .apply { if (fallback != null) putString(KEY_ACTIVE, fallback) }.commit()
        return crashed
    }

    fun hasFailed(name: String): Boolean = name in prefs.getStringSet(KEY_FAILED, emptySet()).orEmpty()

    /** Frees the model on its own thread (after any running generation) without blocking the caller. */
    fun release() {
        executor.execute { unload() }
        executor.shutdown()
    }

    private fun unload() {
        llm?.close()
        llm = null
        loadedFile = null
    }

    companion object {
        /** Prompt + answer tokens. 1280 is the smallest KV-cache size of the Gemma 3 1B .task variants. */
        const val MAX_TOKENS = 1280
        private const val KEY_ACTIVE = "active_model"
        private const val KEY_IN_FLIGHT = "loading_model"
        private const val KEY_FAILED = "crashed_models"
        private val EXTENSIONS = setOf("task", "litertlm")

        /** Keeps the picked file's name (letters, digits, . _ -); rejects anything that is not a model file. */
        fun safeFileName(displayName: String): String {
            val clean = displayName.trim().replace(Regex("""[^A-Za-z0-9._-]"""), "_").trim('.', '_')
            require(clean.substringAfterLast('.', "").lowercase() in EXTENSIONS) {
                "Pick a Gemma model file ending in .task or .litertlm (got \"$displayName\")"
            }
            return clean
        }

        /** "gemma-4-E2B-it.litertlm" -> "Gemma 4 E2B", "gemma3-1b-it-int4.task" -> "Gemma 3 1B". */
        fun friendlyName(fileName: String): String {
            val base = fileName.substringBeforeLast('.').lowercase()
            val m = Regex("""gemma-?(\d+)[-_]?(e?\d+b)""").find(base) ?: return fileName.substringBeforeLast('.')
            return "Gemma ${m.groupValues[1]} ${m.groupValues[2].uppercase()}"
        }
    }
}
