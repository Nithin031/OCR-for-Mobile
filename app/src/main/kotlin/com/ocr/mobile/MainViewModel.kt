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
import com.ocr.core.ImageDecoder
import com.ocr.core.MlKitOcr
import com.ocr.core.OcrPipeline
import com.ocr.core.PipelineResult
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

sealed class AppUiState {
    object Home : AppUiState()
    object Loading : AppUiState()
    data class Result(val bitmap: Bitmap) : AppUiState()
}

data class OcrUi(
    val running: Boolean = false,
    val lines: List<String> = emptyList(),
    val elapsedMs: Long? = null,
    val error: String? = null,
    val lowConfidence: Set<Int> = emptySet(),   // indices into lines
    val fields: List<DocField> = emptyList(),
    val pipelineNote: String = "",
)

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
)

class MainViewModel(application: Application) : AndroidViewModel(application) {

    private val ocr = MlKitOcr()
    private val pipeline = OcrPipeline(ocr)
    private val gemma = GemmaEngine(application)

    private val _uiState = MutableStateFlow<AppUiState>(AppUiState.Home)
    val uiState: StateFlow<AppUiState> = _uiState.asStateFlow()

    private val _ocr = MutableStateFlow(OcrUi())
    val ocrState: StateFlow<OcrUi> = _ocr.asStateFlow()

    private val _ai = MutableStateFlow(AiUi(modelPresent = gemma.hasModel()))
    val aiState: StateFlow<AiUi> = _ai.asStateFlow()

    // One-shot errors shown as Snackbars on the home screen.
    private val _errorMessage = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val errorMessage: SharedFlow<String> = _errorMessage.asSharedFlow()

    fun verifyAssets(context: Context) {
        viewModelScope.launch(Dispatchers.IO) {
            val result = AssetVerifier.verify(context)
            result.okFiles.forEach { Log.d("AssetVerifier", "OK: $it") }
            result.missingFiles.forEach { Log.w("AssetVerifier", "Missing (not used by the ML Kit engine): $it") }
            if (result.corruptedFiles.isNotEmpty()) {
                val details = result.corruptedFiles.entries.joinToString("\n") { (k, v) -> "$k — $v" }
                Log.e("AssetVerifier", "CORRUPTED ASSETS:\n$details")
                _errorMessage.tryEmit("Build error: asset checksum mismatch. See logcat for details.")
            }
        }
    }

    fun loadImageFromUri(context: Context, uri: Uri) {
        viewModelScope.launch(Dispatchers.IO) {
            _uiState.value = AppUiState.Loading
            runCatching { ImageDecoder.decodeBitmap(context, uri) }
                .onSuccess { bitmap ->
                    _uiState.value = AppUiState.Result(bitmap)
                    runOcr(bitmap)
                }
                .onFailure { e ->
                    Log.e("MainViewModel", "Failed to decode image", e)
                    _uiState.value = AppUiState.Home
                    _errorMessage.tryEmit("Failed to load image: ${e.message}")
                }
        }
    }

    private fun runOcr(bitmap: Bitmap) {
        _ocr.value = OcrUi(running = true)
        _ai.update { it.copy(answer = null, question = null, error = null, status = "") }
        viewModelScope.launch(Dispatchers.Default) {
            runCatching { pipeline.run(bitmap) }
                .onSuccess { r ->
                    _ocr.value = OcrUi(
                        lines = r.output.lines.map { it.text },
                        elapsedMs = r.totalMs,
                        lowConfidence = r.lowConfidence,
                        fields = r.fields,
                        pipelineNote = pipelineNote(r),
                    )
                }
                .onFailure { e ->
                    Log.e("MainViewModel", "OCR failed", e)
                    _ocr.value = OcrUi(error = "Text recognition failed: ${e.message}")
                }
        }
    }

    private fun pipelineNote(r: PipelineResult): String {
        val parts = mutableListOf(if (r.passes == 1) "1 pass" else "${r.passes} passes")
        val plan = r.enhancePlan
        if (r.usedEnhanced && plan != null) {
            val how = listOfNotNull(
                "grayscale",
                if (plan.stretches) "contrast boost" else null,
                if (plan.upscale > 1f) "%.1f× upscale".format(plan.upscale) else null,
            )
            parts += "enhanced image used (${how.joinToString(", ")})"
        } else if (r.passes > 1) {
            parts += "original image kept"
        }
        if (r.output.skewDegrees != 0f) parts += "tilt %.1f° corrected in reading order".format(r.output.skewDegrees)
        return parts.joinToString(" · ")
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

    fun ask(question: String) {
        val lines = _ocr.value.lines
        if (question.isBlank() || lines.isEmpty() || _ai.value.busy) return
        val built = DocPrompt.build(lines, question)
        _ai.update {
            it.copy(busy = true, status = "Loading the AI model…", question = question.trim(), answer = null,
                answerSeconds = null, error = null, linesUsed = built.linesUsed, linesTotal = built.linesTotal)
        }
        viewModelScope.launch {
            runCatching {
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
        _uiState.value = AppUiState.Home
        _ocr.value = OcrUi()
        _ai.update { it.copy(answer = null, question = null, error = null, status = "") }
    }

    override fun onCleared() {
        ocr.close()
        gemma.release()
    }
}
