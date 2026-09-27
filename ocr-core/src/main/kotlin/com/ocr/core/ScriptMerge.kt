package com.ocr.core

import kotlin.math.max
import kotlin.math.min

/**
 * Merges a Latin OCR pass with a Devanagari OCR pass of the same page. Bilingual government forms print
 * labels as "जन्म तिथि/Date of Birth"; the Latin recognizer turns the Hindi half into gibberish such as
 * "faf/Date of Birth". Every line the Devanagari pass reads with real Devanagari text replaces the Latin
 * lines it overlaps; all other Latin lines are kept exactly as they were, so English-only pages are unchanged.
 */
object ScriptMerge {

    class Box(val left: Float, val top: Float, val right: Float, val bottom: Float) {
        val area: Float get() = max(0f, right - left) * max(0f, bottom - top)
    }

    /** Share of letters that are Devanagari. */
    fun devanagariShare(text: String): Double {
        val letters = text.count { Character.isLetter(it) }
        if (letters == 0) return 0.0
        return text.count { it in 'ऀ'..'ॿ' }.toDouble() / letters
    }

    /** Overlap relative to the smaller box (a short Latin fragment inside a long Hindi line counts fully). */
    fun overlap(a: Box, b: Box): Double {
        val w = min(a.right, b.right) - max(a.left, b.left)
        val h = min(a.bottom, b.bottom) - max(a.top, b.top)
        if (w <= 0 || h <= 0) return 0.0
        val smaller = min(a.area, b.area)
        return if (smaller <= 0f) 0.0 else (w * h / smaller).toDouble()
    }

    /**
     * Returns the indices to keep: (Latin line indices, Devanagari line indices). A Devanagari line is used
     * when at least [MIN_DEVANAGARI_SHARE] of its letters are Devanagari; it replaces the Latin lines that
     * overlap it by at least [MIN_OVERLAP].
     */
    fun merge(latin: List<Pair<String, Box>>, devanagari: List<Pair<String, Box>>): Pair<List<Int>, List<Int>> {
        val hindi = devanagari.indices.filter { devanagariShare(devanagari[it].first) >= MIN_DEVANAGARI_SHARE }
        if (hindi.isEmpty()) return latin.indices.toList() to emptyList()
        val replaced = BooleanArray(latin.size)
        for (h in hindi) for (l in latin.indices) {
            if (!replaced[l] && overlap(latin[l].second, devanagari[h].second) >= MIN_OVERLAP) replaced[l] = true
        }
        return latin.indices.filter { !replaced[it] } to hindi
    }

    const val MIN_DEVANAGARI_SHARE = 0.3
    const val MIN_OVERLAP = 0.5
}
