package com.ocr.mobile

import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognitionSupport
import android.speech.RecognitionSupportCallback
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.speech.tts.Voice
import android.util.Log
import java.util.Locale

/** Speech language tags per app language (Indian English for English). */
object SpeechLanguages {
    fun tagFor(appLanguage: String): String = when (appLanguage) {
        "hi" -> "hi-IN"
        "kn" -> "kn-IN"
        else -> "en-IN"
    }

    /** Language of a text by its script, like the web app's detect_language: Kannada, Devanagari or English. */
    fun tagForText(text: String): String {
        val kn = text.count { it in 'ಀ'..'೿' }
        val hi = text.count { it in 'ऀ'..'ॿ' }
        val latin = text.count { it in 'A'..'Z' || it in 'a'..'z' }
        return when {
            kn > maxOf(hi.toDouble(), latin * 0.3) -> "kn-IN"
            hi > maxOf(kn.toDouble(), latin * 0.3) -> "hi-IN"
            else -> "en-IN"
        }
    }
}

/**
 * Voice input with Android's on-device recognizer only (Android 12+): audio never leaves the phone.
 * Must be used from the main thread. The recognized text only fills the question box.
 */
class VoiceInput(private val context: Context) {

    sealed class Event {
        data class Partial(val text: String) : Event()
        data class Final(val text: String) : Event()
        /** [needsDownload]: the language pack is missing but Android can download it (once, online). */
        data class Error(val code: Int, val needsDownload: Boolean) : Event()
        object Ready : Event()
        object End : Event()
    }

    enum class Support { INSTALLED, DOWNLOADABLE, UNSUPPORTED, UNKNOWN }

    private var recognizer: SpeechRecognizer? = null

    fun available(): Boolean =
        Build.VERSION.SDK_INT >= Build.VERSION_CODES.S && SpeechRecognizer.isOnDeviceRecognitionAvailable(context)

    private fun intent(tag: String) = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
        putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
        putExtra(RecognizerIntent.EXTRA_LANGUAGE, tag)
        putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true)
        putExtra(RecognizerIntent.EXTRA_MAX_RESULTS, 1)
    }

    fun start(tag: String, onEvent: (Event) -> Unit) {
        if (!available()) {
            onEvent(Event.Error(SpeechRecognizer.ERROR_CLIENT, needsDownload = false))
            return
        }
        stop()
        val r = SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
        recognizer = r
        r.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) = onEvent(Event.Ready)
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onEndOfSpeech() = onEvent(Event.End)
            override fun onEvent(eventType: Int, params: Bundle?) {}
            override fun onPartialResults(partialResults: Bundle?) {
                first(partialResults)?.let { onEvent(Event.Partial(it)) }
            }
            override fun onResults(results: Bundle?) {
                onEvent(Event.Final(first(results).orEmpty()))
            }
            override fun onError(error: Int) {
                Log.w(TAG, "Recognition error $error for $tag")   // error code only, never the audio or text
                val missingPack = error == SpeechRecognizer.ERROR_LANGUAGE_NOT_SUPPORTED ||
                    error == SpeechRecognizer.ERROR_LANGUAGE_UNAVAILABLE
                onEvent(Event.Error(error, needsDownload = missingPack && Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU))
            }
        })
        r.startListening(intent(tag))
    }

    private fun first(results: Bundle?): String? =
        results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull()

    fun stop() {
        recognizer?.let {
            it.cancel()
            it.destroy()
        }
        recognizer = null
    }

    /** Asks Android whether [tag] can be recognized on the device (Android 13+). */
    fun checkSupport(tag: String, onResult: (Support) -> Unit) {
        if (!available() || Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) {
            onResult(if (available()) Support.UNKNOWN else Support.UNSUPPORTED)
            return
        }
        val r = SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
        r.checkRecognitionSupport(intent(tag), context.mainExecutor, object : RecognitionSupportCallback {
            override fun onSupportResult(support: RecognitionSupport) {
                val lang = tag.substringBefore('-')
                fun List<String>.has() = any { it.equals(tag, true) || it.substringBefore('-').equals(lang, true) }
                onResult(
                    when {
                        support.installedOnDeviceLanguages.has() -> Support.INSTALLED
                        support.pendingOnDeviceLanguages.has() || support.supportedOnDeviceLanguages.has() -> Support.DOWNLOADABLE
                        else -> Support.UNSUPPORTED
                    }
                )
                r.destroy()
            }

            override fun onError(error: Int) {
                Log.w(TAG, "checkRecognitionSupport error $error for $tag")
                onResult(Support.UNKNOWN)
                r.destroy()
            }
        })
    }

    /** Starts Android's one-time download of the on-device model for [tag] (needs internet once, Android 13+). */
    fun downloadLanguage(tag: String) {
        if (!available() || Build.VERSION.SDK_INT < Build.VERSION_CODES.TIRAMISU) return
        val r = SpeechRecognizer.createOnDeviceSpeechRecognizer(context)
        r.triggerModelDownload(intent(tag))
        r.destroy()
    }

    companion object {
        private const val TAG = "VoiceInput"
    }
}

/**
 * Read-aloud with Android text-to-speech, restricted to voices that work without a network connection.
 * The language is chosen from the text's script (Kannada, Hindi or English).
 */
class ReadAloud(context: Context, private val onState: (speaking: Boolean) -> Unit) {

    private var ready = false
    // Samsung's default engine has no Hindi or Kannada voices; Google's speech engine has both, so use it
    // when installed and fall back to the phone's default engine otherwise.
    private val engine: String? = GOOGLE_TTS.takeIf { pkg ->
        runCatching { context.packageManager.getPackageInfo(pkg, 0) }.isSuccess
    }

    private val tts: TextToSpeech = TextToSpeech(context.applicationContext, { status ->
        ready = status == TextToSpeech.SUCCESS
        if (!ready) Log.w(TAG, "TextToSpeech init failed: $status (engine ${engine ?: "default"})")
        else Log.i(TAG, "Engine ${engine ?: "default"}; offline voices: " +
            listOf("en-IN", "hi-IN", "kn-IN").joinToString { "$it=${offlineVoice(it)?.name ?: "none"}" })
    }, engine)

    init {
        tts.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) = onState(true)
            override fun onDone(utteranceId: String?) { if (utteranceId == LAST) onState(false) }
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) = onState(false)
            override fun onStop(utteranceId: String?, interrupted: Boolean) = onState(false)
        })
    }

    /** An installed voice for [tag] that needs no network, or null. */
    fun offlineVoice(tag: String): Voice? {
        if (!ready) return null
        val locale = Locale.forLanguageTag(tag)
        val voices = runCatching { tts.voices }.getOrNull().orEmpty().filter { v ->
            v.locale.language == locale.language && !v.isNetworkConnectionRequired &&
                TextToSpeech.Engine.KEY_FEATURE_NOT_INSTALLED !in v.features
        }
        return voices.firstOrNull { it.locale.country == locale.country } ?: voices.firstOrNull()
    }

    /** Speaks [text]; returns false when no offline voice exists for its language. */
    fun speak(text: String): Boolean {
        val tag = SpeechLanguages.tagForText(text)
        val voice = offlineVoice(tag) ?: return false
        tts.stop()
        tts.voice = voice
        val max = TextToSpeech.getMaxSpeechInputLength().coerceAtMost(3000)
        val parts = chunks(text, max)
        parts.forEachIndexed { i, part ->
            tts.speak(part, if (i == 0) TextToSpeech.QUEUE_FLUSH else TextToSpeech.QUEUE_ADD, null,
                if (i == parts.lastIndex) LAST else "part$i")
        }
        return true
    }

    fun stop() {
        tts.stop()
        onState(false)
    }

    fun shutdown() {
        tts.stop()
        tts.shutdown()
    }

    companion object {
        private const val TAG = "ReadAloud"
        private const val LAST = "last"
        private const val GOOGLE_TTS = "com.google.android.tts"

        /** Splits at sentence ends (including the Devanagari danda) so each piece fits the engine's limit. */
        fun chunks(text: String, max: Int): List<String> {
            val out = ArrayList<String>()
            val cur = StringBuilder()
            for (sentence in text.split(Regex("""(?<=[.!?।])\s+"""))) {
                if (cur.isNotEmpty() && cur.length + sentence.length + 1 > max) {
                    out += cur.toString(); cur.clear()
                }
                var s = sentence
                while (s.length > max) { out += s.take(max); s = s.drop(max) }
                if (cur.isNotEmpty()) cur.append(' ')
                cur.append(s)
            }
            if (cur.isNotBlank()) out += cur.toString()
            return out
        }
    }
}
