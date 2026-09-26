package com.ocr.core

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.util.Log
import androidx.exifinterface.media.ExifInterface

object ImageDecoder {

    private const val TAG = "ImageDecoder"
    private const val MAX_LONG_SIDE = 4000

    /**
     * Decodes a bitmap from [uri], applies EXIF rotation, and caps the long
     * side at [MAX_LONG_SIDE] px (logging when the cap kicks in).
     *
     * Must be called on a background thread (file I/O).
     */
    fun decodeBitmap(context: Context, uri: Uri): Bitmap {
        val cr = context.contentResolver

        // Pass 1: read EXIF rotation (separate stream — ExifInterface consumes it).
        val exifMatrix = cr.openInputStream(uri)?.use { stream ->
            exifOrientationMatrix(ExifInterface(stream))
        } ?: Matrix()

        // Pass 2: measure dimensions without allocating pixels.
        val boundsOpts = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        cr.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, boundsOpts) }
        val origW = boundsOpts.outWidth
        val origH = boundsOpts.outHeight
        val longSide = maxOf(origW, origH)

        // Choose inSampleSize so the intermediate allocation stays below 2×MAX_LONG_SIDE.
        var inSampleSize = 1
        while ((longSide / inSampleSize) > MAX_LONG_SIDE * 2) inSampleSize *= 2

        // Pass 3: decode at the chosen sample size.
        val decodeOpts = BitmapFactory.Options().apply {
            this.inSampleSize = inSampleSize
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        var bitmap = cr.openInputStream(uri)?.use {
            BitmapFactory.decodeStream(it, null, decodeOpts)
        } ?: error("Could not open URI: $uri")

        // Precise scale-down to MAX_LONG_SIDE if still needed.
        val decodedLong = maxOf(bitmap.width, bitmap.height)
        if (decodedLong > MAX_LONG_SIDE) {
            Log.i(TAG, "Capping long side ${decodedLong}px → ${MAX_LONG_SIDE}px (original ${longSide}px)")
            val scale = MAX_LONG_SIDE.toFloat() / decodedLong
            val scaled = Bitmap.createScaledBitmap(
                bitmap,
                (bitmap.width * scale).toInt(),
                (bitmap.height * scale).toInt(),
                /* filter = */ true,
            )
            bitmap.recycle()
            bitmap = scaled
        }

        // Apply EXIF rotation / flip.
        if (!exifMatrix.isIdentity) {
            val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, exifMatrix, true)
            bitmap.recycle()
            bitmap = rotated
        }

        return bitmap
    }

    private fun exifOrientationMatrix(exif: ExifInterface): Matrix {
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL,
        )
        return Matrix().apply {
            when (orientation) {
                ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> postScale(-1f, 1f)
                ExifInterface.ORIENTATION_ROTATE_180 -> postRotate(180f)
                ExifInterface.ORIENTATION_FLIP_VERTICAL -> postScale(1f, -1f)
                ExifInterface.ORIENTATION_TRANSPOSE -> { postScale(-1f, 1f); postRotate(-90f) }
                ExifInterface.ORIENTATION_ROTATE_90 -> postRotate(90f)
                ExifInterface.ORIENTATION_TRANSVERSE -> { postScale(-1f, 1f); postRotate(90f) }
                ExifInterface.ORIENTATION_ROTATE_270 -> postRotate(270f)
                // ORIENTATION_NORMAL and ORIENTATION_UNDEFINED: identity
            }
        }
    }
}
