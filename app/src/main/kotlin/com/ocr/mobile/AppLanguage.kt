package com.ocr.mobile

import android.content.Context
import android.content.res.Configuration
import java.util.Locale

/**
 * The app's display language (English, Hindi, Kannada), chosen in the app and stored on the phone.
 * English is the default, so the app looks exactly as before unless the user picks another language.
 * Only the app's own labels change; recognized text, engine names and timings are not translated.
 */
object AppLanguage {

    data class Option(val code: String, val label: String)

    val OPTIONS = listOf(
        Option("en", "English"),
        Option("hi", "हिन्दी"),
        Option("kn", "ಕನ್ನಡ"),
    )

    private const val PREFS = "app_language"
    private const val KEY = "code"
    const val DEFAULT = "en"

    fun get(context: Context): String {
        val code = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getString(KEY, DEFAULT) ?: DEFAULT
        return if (OPTIONS.any { it.code == code }) code else DEFAULT
    }

    fun set(context: Context, code: String) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putString(KEY, code).apply()
    }

    /** A context whose resources use the chosen language. */
    fun wrap(base: Context): Context {
        val locale = Locale(get(base))
        val config = Configuration(base.resources.configuration)
        config.setLocale(locale)
        return base.createConfigurationContext(config)
    }
}
