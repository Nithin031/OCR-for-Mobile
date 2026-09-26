package com.ocr.mobile

import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.ocr.core.AssetVerifier
import com.ocr.core.ImageDecoder
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asSharedFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

sealed class AppUiState {
    object Home : AppUiState()
    object Loading : AppUiState()
    data class Result(val bitmap: Bitmap) : AppUiState()
}

class MainViewModel : ViewModel() {

    private val _uiState = MutableStateFlow<AppUiState>(AppUiState.Home)
    val uiState: StateFlow<AppUiState> = _uiState.asStateFlow()

    // One-shot errors shown as Snackbars on the home screen.
    private val _errorMessage = MutableSharedFlow<String>(extraBufferCapacity = 1)
    val errorMessage: SharedFlow<String> = _errorMessage.asSharedFlow()

    fun verifyAssets(context: Context) {
        viewModelScope.launch(Dispatchers.IO) {
            val result = AssetVerifier.verify(context)
            result.okFiles.forEach { Log.d("AssetVerifier", "OK: $it") }
            result.missingFiles.forEach {
                Log.w("AssetVerifier", "Missing (will fail OCR when needed): $it")
            }
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
                .onSuccess { _uiState.value = AppUiState.Result(it) }
                .onFailure { e ->
                    Log.e("MainViewModel", "Failed to decode image", e)
                    _uiState.value = AppUiState.Home
                    _errorMessage.tryEmit("Failed to load image: ${e.message}")
                }
        }
    }

    fun goHome() {
        _uiState.value = AppUiState.Home
    }
}
