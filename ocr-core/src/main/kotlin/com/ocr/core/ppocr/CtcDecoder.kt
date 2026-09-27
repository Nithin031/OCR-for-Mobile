package com.ocr.core.ppocr

/**
 * Greedy CTC decoding, mirroring PaddleX CTCLabelDecode (text_recognition/processors.py):
 * argmax per timestep, drop repeats of the previous index, drop the blank (index 0);
 * index i in 1..dict.size maps to dict[i - 1], the last index is the appended space.
 * Confidence is the mean of the max probability over the kept timesteps (0 if none); the model
 * output is used as-is, exactly like PaddleX (no extra softmax).
 */
class CtcDecoder(private val dictionary: List<String>, private val blankIndex: Int, private val spaceIndex: Int) {

    val classes: Int get() = dictionary.size + 2

    /** [probs] holds one sequence of [steps] x [classes] values starting at [offset]. */
    fun decode(probs: FloatArray, offset: Int, steps: Int): Pair<String, Float> {
        val text = StringBuilder()
        var confSum = 0.0
        var kept = 0
        var prev = -1
        for (t in 0 until steps) {
            val base = offset + t * classes
            var best = 0
            var bestP = probs[base]
            for (c in 1 until classes) {
                val p = probs[base + c]
                if (p > bestP) {
                    bestP = p
                    best = c
                }
            }
            if (best != prev && best != blankIndex) {
                text.append(if (best == spaceIndex) " " else dictionary[best - 1])
                confSum += bestP
                kept++
            }
            prev = best
        }
        return text.toString() to if (kept == 0) 0f else (confSum / kept).toFloat()
    }
}
