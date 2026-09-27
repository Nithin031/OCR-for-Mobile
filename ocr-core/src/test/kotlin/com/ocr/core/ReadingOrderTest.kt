package com.ocr.core

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.tan

class ReadingOrderTest {

    private data class L(val name: String, val left: Float, val width: Float, val cy: Float, val h: Float = 20f)

    private fun geo(l: L) = ReadingOrder.Geo(l.left, l.left + l.width / 2, l.cy, l.h)

    private fun order(lines: List<L>, skew: Float = 0f) =
        ReadingOrder.rows(lines, skew, ::geo).map { row -> row.map { it.name } }

    @Test
    fun rowsTopToBottomThenLeftToRight() {
        val lines = listOf(
            L("value", 300f, 200f, 104f),
            L("label", 20f, 200f, 100f),
            L("next", 20f, 200f, 160f),
        )
        assertEquals(listOf(listOf("label", "value"), listOf("next")), order(lines))
    }

    @Test
    fun skewedPageKeepsRowsTogether() {
        // Page tilted 8 degrees clockwise: text further right sits lower.
        val t = tan(Math.toRadians(8.0)).toFloat()
        fun at(name: String, left: Float, baseY: Float) = L(name, left, 200f, baseY + (left + 100f) * t)
        val lines = listOf(
            at("a1", 20f, 100f), at("a2", 900f, 100f),
            at("b1", 20f, 140f), at("b2", 900f, 140f),
        )
        // Without skew correction the rows get mixed up...
        assertEquals(listOf(listOf("a1"), listOf("b1"), listOf("a2"), listOf("b2")), order(lines))
        // ...with it they are read as two rows.
        assertEquals(listOf(listOf("a1", "a2"), listOf("b1", "b2")), order(lines, 8f))
    }

    @Test
    fun skewEstimateIgnoresNoiseAndOutliers() {
        assertEquals(0f, ReadingOrder.estimateSkew(listOf(0.2f, -0.3f, 0.1f)), 0f)
        assertEquals(0f, ReadingOrder.estimateSkew(listOf(5f, 5f)), 0f) // too few lines
        assertEquals(4f, ReadingOrder.estimateSkew(listOf(3.8f, 4f, 4.2f, 90f)), 0f)
    }
}
