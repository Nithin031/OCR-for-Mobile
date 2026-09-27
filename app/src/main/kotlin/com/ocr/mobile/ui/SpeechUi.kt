package com.ocr.mobile.ui

import android.Manifest
import android.content.pm.PackageManager
import android.speech.SpeechRecognizer
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.core.content.ContextCompat
import com.ocr.mobile.AppLanguage
import com.ocr.mobile.R
import com.ocr.mobile.ReadAloud
import com.ocr.mobile.SpeechLanguages
import com.ocr.mobile.VoiceInput

private fun languageLabel(tag: String): String {
    val code = tag.substringBefore('-')
    return AppLanguage.OPTIONS.firstOrNull { it.code == code }?.label ?: tag
}

/** State of voice input for the question box. Speech only fills the box; it never sends the question. */
class VoiceQuestionState {
    var listening by mutableStateOf(false)
    var message by mutableStateOf<String?>(null)
    var offerDownloadFor by mutableStateOf<String?>(null)
    var unsupported by mutableStateOf(false)
}

@Composable
fun rememberVoiceQuestion(): Pair<VoiceInput, VoiceQuestionState> {
    val context = LocalContext.current
    val voice = remember { VoiceInput(context) }
    val state = remember { VoiceQuestionState() }
    DisposableEffect(Unit) { onDispose { voice.stop() } }
    return voice to state
}

/** Microphone button for the question field's trailing icon. */
@Composable
fun VoiceQuestionButton(
    voice: VoiceInput,
    state: VoiceQuestionState,
    enabled: Boolean,
    onText: (String) -> Unit,
    wide: Boolean = false,
) {
    val context = LocalContext.current
    val tag = SpeechLanguages.tagFor(AppLanguage.get(context))
    val lang = languageLabel(tag)
    val sListening = stringResource(R.string.listening, lang)
    val sMissing = stringResource(R.string.voice_missing_pack, lang)
    val sUnavailable = stringResource(R.string.voice_unavailable)
    val sNoMatch = stringResource(R.string.voice_no_match)
    val sPermission = stringResource(R.string.mic_permission)
    val sNotOffline = stringResource(R.string.voice_not_offline, lang)

    fun listen() {
        if (!voice.available()) {
            state.message = sUnavailable
            return
        }
        if (state.unsupported) {        // e.g. Kannada: Android has no offline recognizer for it on this phone
            state.message = sNotOffline
            return
        }
        state.offerDownloadFor = null
        state.message = sListening
        state.listening = true
        voice.start(tag) { event ->
            when (event) {
                is VoiceInput.Event.Partial -> onText(event.text)
                is VoiceInput.Event.Final -> {
                    state.listening = false
                    if (event.text.isBlank()) state.message = sNoMatch else { onText(event.text); state.message = null }
                }
                is VoiceInput.Event.Error -> {
                    state.listening = false
                    state.message = when {
                        event.needsDownload -> sMissing.also { state.offerDownloadFor = tag }
                        event.code == SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED ||
                            event.code == SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE -> sMissing
                        event.code == SpeechRecognizer.ERROR_NO_MATCH ||
                            event.code == SpeechRecognizer.ERROR_SPEECH_TIMEOUT -> sNoMatch
                        else -> sNoMatch
                    }
                }
                VoiceInput.Event.Ready, VoiceInput.Event.End -> {}
            }
        }
    }

    // Check once whether this language can be recognized offline, so a missing pack is offered for
    // download up front instead of failing in the middle of a demo.
    LaunchedEffect(tag) {
        voice.checkSupport(tag) { support ->
            Log.i("VoiceInput", "On-device recognition for $tag: $support")
            state.unsupported = support == VoiceInput.Support.UNSUPPORTED
            if (support == VoiceInput.Support.DOWNLOADABLE && !state.listening) {
                state.message = sMissing
                state.offerDownloadFor = tag
            }
        }
    }

    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted) listen() else state.message = sPermission
    }

    val onClick = {
        when {
            state.listening -> { voice.stop(); state.listening = false; state.message = null }
            ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO) ==
                PackageManager.PERMISSION_GRANTED -> listen()
            else -> permission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }
    // Material core icons have no microphone; emoji keep the APK free of the extended icon set.
    val icon = if (state.listening) "⏹" else "🎤"
    if (wide) {
        OutlinedButton(enabled = enabled, onClick = onClick) {
            Text("$icon " + stringResource(if (state.listening) R.string.stop_listening else R.string.voice_input))
        }
    } else {
        IconButton(enabled = enabled, onClick = onClick) {
            Text(icon, style = MaterialTheme.typography.titleMedium)
        }
    }
}

/** Status line under the question field: listening / missing language pack (with download) / errors. */
@Composable
fun VoiceQuestionStatus(voice: VoiceInput, state: VoiceQuestionState) {
    val message = state.message ?: return
    val sDownloading = stringResource(R.string.voice_downloading)
    Column {
        Text(message, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
        val tag = state.offerDownloadFor
        if (tag != null) {
            TextButton(onClick = {
                voice.downloadLanguage(tag)
                state.offerDownloadFor = null
                state.message = sDownloading
            }) { Text(stringResource(R.string.voice_download)) }
        }
    }
}

/** Shared read-aloud engine for the result screen, shut down when the screen goes away. */
class ReadAloudState {
    var speaking by mutableStateOf(false)
    var message by mutableStateOf<String?>(null)
    var current by mutableStateOf<String?>(null)   // which text is being read (button shows Stop there)
}

@Composable
fun rememberReadAloud(): Pair<ReadAloud, ReadAloudState> {
    val context = LocalContext.current
    val state = remember { ReadAloudState() }
    val tts = remember { ReadAloud(context) { speaking -> state.speaking = speaking; if (!speaking) state.current = null } }
    DisposableEffect(Unit) { onDispose { tts.shutdown() } }
    return tts to state
}

/** "Read aloud" / "Stop reading" for [text], in the voice matching the text's script (offline voices only). */
@Composable
fun ReadAloudButton(text: String, key: String, tts: ReadAloud, state: ReadAloudState) {
    val lang = languageLabel(SpeechLanguages.tagForText(text))
    val sNoVoice = stringResource(R.string.no_offline_voice, lang)
    val reading = state.speaking && state.current == key
    Column {
        TextButton(onClick = {
            if (reading) {
                tts.stop()
            } else {
                state.message = null
                if (tts.speak(text)) state.current = key else state.message = sNoVoice
            }
        }) { Text(if (reading) "⏹ " + stringResource(R.string.stop_reading) else "🔊 " + stringResource(R.string.read_aloud)) }
        if (state.message != null && !reading) {
            Text(state.message.orEmpty(), style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
        }
    }
}
