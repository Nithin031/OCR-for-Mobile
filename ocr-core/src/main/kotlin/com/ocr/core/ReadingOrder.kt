package com.ocr.core

import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/** Pure geometry for putting OCR lines in reading order (no Android types, so it is unit-testable). */
internal object ReadingOrder {

    /** Axis-aligned line box, reduced to what ordering needs. */
    class Geo(val left: Float, val centreX: Float, val centreY: Float, val height: Float)

    /** Page skew from ML Kit's per-line angles (degrees, clockwise positive); 0 when there is no clear skew. */
    fun estimateSkew(angles: List<Float>): Float {
        val usable = angles.filter { abs(it) <= MAX_SKEW }.sorted()
        if (usable.size < MIN_LINES_FOR_SKEW) return 0f
        val median = usable[usable.size / 2]
        return if (abs(median) < MIN_SKEW) 0f else median
    }

    /**
     * Groups items into rows, top to bottom, each row left to right. Coordinates are first rotated by
     * -[skewDeg] so the lines of a tilted page are horizontal. A line joins the current row when its
     * centre is within 0.5 x the median line height of the row's average centre.
     */
    fun <T> rows(items: List<T>, skewDeg: Float, geo: (T) -> Geo): List<List<T>> {
        if (items.isEmpty()) return emptyList()
        val rad = Math.toRadians(skewDeg.toDouble())
        val c = cos(rad).toFloat()
        val s = sin(rad).toFloat()

        class P(val item: T, val y: Float, val x: Float)
        val points = items.map { item ->
            val g = geo(item)
            P(item, y = g.centreY * c - g.centreX * s, x = g.left * c + g.centreY * s)
        }
        val heights = items.map { geo(it).height }.sorted()
        val tolerance = heights[heights.size / 2] * 0.5f

        val rows = mutableListOf<MutableList<P>>()
        var rowCentre = 0f
        for (p in points.sortedBy { it.y }) {
            if (rows.isEmpty() || p.y - rowCentre > tolerance) {
                rows.add(mutableListOf(p))
                rowCentre = p.y
            } else {
                val row = rows.last()
                row.add(p)
                rowCentre = row.map { it.y }.average().toFloat()
            }
        }
        return rows.map { row -> row.sortedBy { it.x }.map { it.item } }
    }

    private const val MAX_SKEW = 30f
    private const val MIN_SKEW = 0.5f
    private const val MIN_LINES_FOR_SKEW = 3
}
