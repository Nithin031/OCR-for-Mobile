package com.ocr.core

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.ColorMatrix
import android.graphics.ColorMatrixColorFilter
import android.graphics.Paint
import android.graphics.Rect
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

/**
 * Makes a grayscale, contrast-stretched and (for small images) upscaled copy of a page for a second OCR
 * pass. Faded photocopies and low-resolution gallery images are where ML Kit loses the most text.
 */
object ImageEnhancer {

    /** [upscale] >= 1; luminance in [low]..[high] is stretched to 0..255. */
    data class Plan(val upscale: Float, val low: Int, val high: Int) {
        val stretches: Boolean get() = high - low < GOOD_CONTRAST
    }

    /** Returns null when the image is already large enough and has good contrast. */
    fun plan(src: Bitmap): Plan? {
        val shortSide = min(src.width, src.height)
        val longSide = max(src.width, src.height)
        var up = if (shortSide < TARGET_SHORT_SIDE) min(MAX_UPSCALE, TARGET_SHORT_SIDE.toFloat() / shortSide) else 1f
        up = min(up, MAX_LONG_SIDE.toFloat() / longSide)
        if (up < MIN_USEFUL_UPSCALE) up = 1f

        val (low, high) = luminancePercentiles(src)
        val plan = Plan(up, low, max(high, low + 1))
        return if (plan.upscale == 1f && !plan.stretches) null else plan
    }

    fun apply(src: Bitmap, plan: Plan): Bitmap {
        val w = (src.width * plan.upscale).roundToInt()
        val h = (src.height * plan.upscale).roundToInt()
        val out = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val (low, high) = if (plan.stretches) plan.low to plan.high else 0 to 255
        val scale = 255f / (high - low)
        val offset = -low * scale
        val r = 0.299f * scale
        val g = 0.587f * scale
        val b = 0.114f * scale
        val matrix = ColorMatrix(
            floatArrayOf(
                r, g, b, 0f, offset,
                r, g, b, 0f, offset,
                r, g, b, 0f, offset,
                0f, 0f, 0f, 1f, 0f,
            )
        )
        val paint = Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG).apply {
            colorFilter = ColorMatrixColorFilter(matrix)
        }
        Canvas(out).apply {
            drawColor(Color.WHITE)
            drawBitmap(src, null, Rect(0, 0, w, h), paint)
        }
        return out
    }

    /** 0.5th and 99.5th luminance percentiles over a sampled grid of about 250k pixels. */
    private fun luminancePercentiles(src: Bitmap): Pair<Int, Int> {
        val step = max(1, sqrt(src.width.toDouble() * src.height / SAMPLE_PIXELS).toInt())
        val hist = IntArray(256)
        val row = IntArray(src.width)
        var count = 0
        var y = 0
        while (y < src.height) {
            src.getPixels(row, 0, src.width, 0, y, src.width, 1)
            var x = 0
            while (x < src.width) {
                val p = row[x]
                val lum = (299 * ((p shr 16) and 0xFF) + 587 * ((p shr 8) and 0xFF) + 114 * (p and 0xFF)) / 1000
                hist[lum]++
                count++
                x += step
            }
            y += step
        }
        return percentile(hist, count, 0.005) to percentile(hist, count, 0.995)
    }

    private fun percentile(hist: IntArray, count: Int, q: Double): Int {
        val target = (count * q).toLong()
        var acc = 0L
        for (i in hist.indices) {
            acc += hist[i]
            if (acc > target) return i
        }
        return 255
    }

    /** ML Kit wants roughly 16+ px per character; 1200 px on the short side covers A4 body text. */
    private const val TARGET_SHORT_SIDE = 1200
    private const val MAX_UPSCALE = 2f
    private const val MIN_USEFUL_UPSCALE = 1.15f
    private const val MAX_LONG_SIDE = 4000
    private const val GOOD_CONTRAST = 160
    private const val SAMPLE_PIXELS = 250_000.0
}
