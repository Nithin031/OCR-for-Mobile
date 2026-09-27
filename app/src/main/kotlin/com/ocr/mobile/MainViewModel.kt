package com.ocr.mobile

import android.app.Application
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.ocr.core.AssetVerifier
import com.ocr.core.DocField
import com.ocr.core.EngineRegistry
import com.ocr.core.ImageDecoder
import com.ocr.core.OcrEngine
import com.ocr.core.OcrLine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import com.ocr.mobile.kag.KagRetriever
import com.ocr.mobile.kag.KagStore
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

sealed class AppUiState {
    object Home : AppUiState()
    object Loading : AppUiState()
    data class Result(val bitmap: Bitmap) : AppUiState()
}

data class OcrUi(
    val engineId: String = EngineRegistry.DEFAULT_ID,
    val engineName: String = "",
    val running: Boolean = false,
    val stage: String = "",
    val ocrLines: List<OcrLine> = emptyList(),
    val elapsedMs: Long? = null,
    val timings: List<Pair<String, Long>> = emptyList(),
    val error: String? = null,
    val lowConfidence: Set<Int> = emptySet(),   // indices into ocrLines
    val fields: List<DocField> = emptyList(),
    val pipelineNote: String = "",
) {
    val lines: List<String> get() = ocrLines.map { it.text }
}

data class AiUi(
    val modelPresent: Boolean = false,
    val importProgress: Float? = null,   // non-null while copying the model file
    val busy: Boolean = false,
    val status: String = "",
    val question: String? = null,
    val answer: String? = null,
    val answerSeconds: Double? = null,
    val linesUsed: Int = 0,
    val linesTotal: Int = 0,
    val error: String? = null,
    // Offline scheme knowledge base (KAG) from the web app.
    val kagReady: Boolean = false,
    val kagStatus: String = "",
    val useKnowledge: Boolean = true,
    val sources: List<KagSource> = emptyList(),
)

data class KagSource(val id: String, val title: String, val url: String?)

const val GOLDEN_RUNNING = "Running golden check…"

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val gemma = GemmaEngine(application)

    // Engines are created and initialized lazily, once, off the main thread.
    private val engines = mutableMapOf<String, OcrEngine>()
    private val enginesLock = Mutex()
    private var ocrJob: Job? = null

    private val _uiState = MutableStateFlow<AppUiState>(AppUiState.Home)
    val uiState: StateFlow<AppUiState> = _uiState.asStateFlow()

    private val _engineId = MutableStateFlow(EngineRegistry.DEFAULT_ID)
    val engineId: StateFlow<String> = _engineId.asStateFlow()

    private val _ocr = MutableStateFlow(OcrUi())
    val ocrState: StateFlow<OcrUi> = _ocr.asStateFlow()

    private val _ai = MutableStateFlow(AiUi(modelPresent = gemma.hasModel()))
    val aiState: StateFlow<AiUi> = _ai.asStateFlow()

    // One-shot errors shown as Snackbars on the home screen.
    private val _errorMessage = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val errorMessage: SharedFlow<String> = _errorMessage.asSharedFlow()

    @Volatile private var kag: KagRetriever? = null

    init {
        // Load every engine's models in the background at app start (PP-OCR takes a few seconds).
        EngineRegistry.entries.forEach { warmUp(it.id) }
        loadKnowledge()
    }


    fun verifyAssets(context: Context) {
        viewModelScope.launch(Dispatchers.IO) {
            val result = AssetVerifier.verify(context)
            result.okFiles.forEach { Log.d("AssetVerifier", "OK: $it") }
            result.missingFiles.forEach { Log.w("AssetVerifier", "Missing: $it") }
            if (result.corruptedFiles.isNotEmpty()) {
                val details = result.corruptedFiles.entries.joinToString("\n") { (k, v) -> "$k — $v" }
                Log.e("AssetVerifier", "CORRUPTED ASSETS:\n$details")
                _errorMessage.tryEmit("Build error: asset checksum mismatch. See logcat for details.")
            }
        }
    }

    fun selectEngine(id: String) {
        _engineId.value = EngineRegistry.find(id).id
    }

    /** Starts loading an engine's models in the background so the first scan does not wait for it. */
    fun warmUp(id: String) {
        viewModelScope.launch(Dispatchers.Default) {
            runCatching { engine(id) }.onFailure { e -> Log.e("MainViewModel", "Engine $id failed to initialize", e) }
        }
    }

    private suspend fun engine(id: String): OcrEngine = enginesLock.withLock {
        engines[id] ?: run {
            val entry = EngineRegistry.find(id)
            val start = System.currentTimeMillis()
            val e = entry.create(getApplication())
            try {
                e.initialize()
            } catch (t: Throwable) {
                e.close()
                throw t
            }
            Log.i("MainViewModel", "Engine ${entry.id} ready in ${System.currentTimeMillis() - start} ms")
            engines[entry.id] = e
            e
        }
    }

    fun loadImageFromUri(context: Context, uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = AppUiState.Loading
            val start = System.currentTimeMillis()
            runCatching { ImageDecoder.decodeBitmap(context, uri) }
                .onSuccess { bitmap ->
                    val decodeMs = System.currentTimeMillis() - start
                    _uiState.value = AppUiState.Result(bitmap)
                    runOcr(bitmap, _engineId.value, decodeMs)
                }
                .onFailure { e ->
                    Log.e("MainViewModel", "Failed to decode image", e)
                    _uiState.value = AppUiState.Home
                    _errorMessage.tryEmit("Failed to load image: ${e.message}")
                }
        }
    }

    /** Runs another engine on the image currently shown (e.g. "Run with ML Kit" after a failure). */
    fun rerun(engineId: String) {
        val state = _uiState.value as? AppUiState.Result ?: return
        selectEngine(engineId)
        runOcr(state.bitmap, _engineId.value, decodeMs = null)
    }

    private fun runOcr(bitmap: Bitmap, engineId: String, decodeMs: Long?) {
        val entry = EngineRegistry.find(engineId)
        ocrJob?.cancel()
        _ocr.value = OcrUi(engineId = entry.id, engineName = entry.displayName, running = true, stage = "Loading engine…")
        _ai.update { it.copy(answer = null, question = null, error = null, status = "") }
        ocrJob = viewModelScope.launch(Dispatchers.Default) {
            runCatching {
                val engine = engine(entry.id)
                engine.recognize(bitmap) { stage -> _ocr.update { it.copy(stage = stage) } }
            }.onSuccess { r ->
                val timings = listOfNotNull(decodeMs?.let { "Decode image" to it }) + r.timings
                val skew = r.output.skewDegrees
                val note = listOfNotNull(
                    r.note.ifEmpty { null },
                    if (skew != 0f) "tilt %.1f° corrected in reading order".format(skew) else null,
                ).joinToString(" · ")
                _ocr.value = OcrUi(
                    engineId = entry.id,
                    engineName = entry.displayName,
                    ocrLines = r.output.lines,
                    elapsedMs = r.totalMs,
                    timings = timings,
                    lowConfidence = r.lowConfidence,
                    fields = r.fields,
                    pipelineNote = note,
                )
            }.onFailure { e ->
                Log.e("MainViewModel", "OCR failed with ${entry.id}", e)
                _ocr.value = OcrUi(
                    engineId = entry.id,
                    engineName = entry.displayName,
                    error = "${entry.displayName} failed: ${e.message ?: e.javaClass.simpleName}",
                )
            }
        }
    }

    // Debug-only golden smoke check (sample assets exist only in debug builds).
    val goldenAvailable: Boolean = GoldenCheck.available(application)
    private val _golden = MutableStateFlow<String?>(null)
    val golden: StateFlow<String?> = _golden.asStateFlow()

    fun runGoldenCheck() {
        if (_golden.value == GOLDEN_RUNNING) return
        _golden.value = GOLDEN_RUNNING
        viewModelScope.launch(Dispatchers.Default) {
            _golden.value = runCatching {
                val engine = engine(_engineId.value)
                val (report, file) = GoldenCheck.run(getApplication(), engine)
                "Report saved to ${file.absolutePath}\n\n$report"
            }.getOrElse { e ->
                Log.e("MainViewModel", "Golden check failed", e)
                "Golden check failed: ${e.message ?: e.javaClass.simpleName}"
            }
        }
    }

    fun dismissGolden() {
        if (_golden.value != GOLDEN_RUNNING) _golden.value = null
    }

    fun importModel(uri: Uri) {
        if (_ai.value.busy || _ai.value.importProgress != null) return
        _ai.update { it.copy(importProgress = 0f, error = null) }
        viewModelScope.launch {
            runCatching { gemma.importModel(uri) { p -> _ai.update { s -> s.copy(importProgress = p) } } }
                .onSuccess { _ai.update { it.copy(importProgress = null, modelPresent = gemma.hasModel()) } }
                .onFailure { e ->
                    Log.e("MainViewModel", "Model import failed", e)
                    _ai.update { it.copy(importProgress = null, modelPresent = gemma.hasModel(), error = "Import failed: ${e.message}") }
                }
        }
    }

    fun setUseKnowledge(on: Boolean) {
        _ai.update { it.copy(useKnowledge = on) }
    }

    private fun loadKnowledge() {
        val app = getApplication<Application>()
        if (!KagStore.available(app)) return
        _ai.update { it.copy(kagStatus = "Loading scheme knowledge base…") }
        viewModelScope.launch(Dispatchers.IO) {
            runCatching { KagStore.load(app) }
                .onSuccess { (retriever, info) ->
                    kag = retriever
                    _ai.update {
                        it.copy(kagReady = true, kagStatus = "${info.documents} documents · ${info.chunks} passages · " +
                            "${info.schemes} schemes in graph")
                    }
                }
                .onFailure { e ->
                    Log.e("MainViewModel", "KAG database failed to load", e)
                    _ai.update { it.copy(kagReady = false, useKnowledge = false, kagStatus = "Knowledge base unavailable: ${e.message}") }
                }
        }
    }

    fun ask(question: String) {
        val lines = _ocr.value.lines
        if (question.isBlank() || lines.isEmpty() || _ai.value.busy) return
        val retriever = kag.takeIf { _ai.value.useKnowledge }
        _ai.update {
            it.copy(busy = true, status = if (retriever != null) "Searching the scheme knowledge base…" else "Loading the AI model…",
                question = question.trim(), answer = null, answerSeconds = null, error = null, sources = emptyList())
        }
        viewModelScope.launch {
            runCatching {
                // KAG retrieval (keyword + graph) runs off the main thread; the OCR text names the schemes in context.
                val (built, sources) = withContext(Dispatchers.Default) {
                    if (retriever == null) {
                        DocPrompt.build(lines, question) to emptyList()
                    } else {
                        val ocrText = lines.joinToString("\n")
                        val q = retriever.understand(question, contextSchemes = retriever.graph.detectSchemeMentions(ocrText))
                        val result = retriever.retrieve(q)
                        val (context, used) = KagRetriever.buildContext(result, DocPrompt.MAX_KAG_CHARS)
                        Log.i("MainViewModel", "KAG intent=${q.intent} schemes=${result.schemes.map { it.code }} " +
                            "chunks=${result.chunks.size} facts=${result.facts.size} used=${used.size}")
                        DocPrompt.buildWithKnowledge(lines, question, context) to used
                    }
                }
                _ai.update {
                    it.copy(status = "Loading the AI model…", linesUsed = built.linesUsed, linesTotal = built.linesTotal,
                        sources = sources.map { s -> KagSource(s.id, s.title, s.url) })
                }
                gemma.load()
                _ai.update { it.copy(status = "Thinking… (on this phone, can take up to a minute)") }
                val start = System.currentTimeMillis()
                val answer = gemma.generate(built.prompt)
                answer to (System.currentTimeMillis() - start) / 1000.0
            }.onSuccess { (answer, seconds) ->
                _ai.update { it.copy(busy = false, status = "", answer = answer.trim(), answerSeconds = seconds) }
            }.onFailure { e ->
                Log.e("MainViewModel", "Assistant failed", e)
                _ai.update { it.copy(busy = false, status = "", error = "Assistant failed: ${e.message}") }
            }
        }
    }

    fun goHome() {
        ocrJob?.cancel()
        _uiState.value = AppUiState.Home
        _ocr.value = OcrUi()
        _ai.update { it.copy(answer = null, question = null, error = null, status = "") }
    }

    override fun onCleared() {
        engines.values.forEach { runCatching { it.close() } }
        engines.clear()
        gemma.release()
    }
}
