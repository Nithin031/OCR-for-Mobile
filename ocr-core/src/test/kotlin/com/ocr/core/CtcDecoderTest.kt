package com.ocr.core

import com.ocr.core.ppocr.CtcDecoder
import org.junit.Assert.assertEquals
import org.junit.Test

class CtcDecoderTest {

    // Classes: 0 blank, 1 "a", 2 "b", 3 space.
    private val decoder = CtcDecoder(listOf("a", "b"), blankIndex = 0, spaceIndex = 3)

    private fun steps(vararg best: Int, p: Float = 0.9f): FloatArray {
        val out = FloatArray(best.size * 4) { 0.01f }
        best.forEachIndexed { t, c -> out[t * 4 + c] = p }
        return out
    }

    @Test
    fun collapsesRepeatsAndDropsBlank() {
        val (text, conf) = decoder.decode(steps(1, 1, 0, 1, 2, 2, 3, 2), 0, 8)
        assertEquals("aab b", text)
        assertEquals(0.9f, conf, 1e-6f)
    }

    @Test
    fun emptySequenceHasZeroConfidence() {
        assertEquals("" to 0f, decoder.decode(steps(0, 0, 0), 0, 3))
    }

    @Test
    fun readsAtOffset() {
        val two = steps(0, 0) + steps(2, 1)
        assertEquals("ba", decoder.decode(two, 8, 2).first)
    }
}
