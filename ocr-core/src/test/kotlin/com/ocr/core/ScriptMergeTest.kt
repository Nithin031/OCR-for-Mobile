package com.ocr.core

import com.ocr.core.ScriptMerge.Box
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ScriptMergeTest {

    private fun line(text: String, l: Float, t: Float, r: Float, b: Float) = text to Box(l, t, r, b)

    @Test
    fun englishOnlyPageIsUnchanged() {
        val latin = listOf(line("Name of the member", 10f, 10f, 200f, 30f), line("Date of Birth", 10f, 40f, 150f, 60f))
        // The Devanagari recognizer also reads Latin; without Devanagari letters nothing is replaced.
        val deva = listOf(line("Name of the member", 10f, 10f, 200f, 30f))
        assertEquals(listOf(0, 1) to emptyList<Int>(), ScriptMerge.merge(latin, deva))
    }

    @Test
    fun garbledHindiLabelIsReplacedByTheDevanagariLine() {
        val latin = listOf(
            line("faf/Date of Birth", 10f, 40f, 180f, 60f),          // Hindi half read as Latin gibberish
            line("EMPLOYEES' PROVIDENT FUNDS SCHEME", 10f, 5f, 300f, 25f),
        )
        val deva = listOf(line("जन्म तिथि/Date of Birth", 8f, 38f, 185f, 62f))
        val (keepLatin, useHindi) = ScriptMerge.merge(latin, deva)
        assertEquals(listOf(1), keepLatin)
        assertEquals(listOf(0), useHindi)
    }

    @Test
    fun hindiLineWithoutALatinCounterpartIsAdded() {
        val latin = listOf(line("Form 19", 10f, 5f, 80f, 20f))
        val deva = listOf(line("कर्मचारी भविष्य निधि योजना", 10f, 100f, 300f, 120f))
        assertEquals(listOf(0) to listOf(0), ScriptMerge.merge(latin, deva))
    }

    @Test
    fun devanagariShareIgnoresMostlyLatinLines() {
        assertTrue(ScriptMerge.devanagariShare("जन्म तिथि/Date of Birth") >= ScriptMerge.MIN_DEVANAGARI_SHARE)
        assertTrue(ScriptMerge.devanagariShare("www.epfindia.gov.in") < ScriptMerge.MIN_DEVANAGARI_SHARE)
        assertEquals(0.0, ScriptMerge.devanagariShare("15/15"), 0.0)
    }
}
