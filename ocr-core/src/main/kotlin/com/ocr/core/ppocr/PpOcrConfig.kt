package com.ocr.core.ppocr

import org.json.JSONObject

/**
 * Every pre/post-processing value used by [PpOcrV6Engine], read from the Phase-0 assets:
 * models/preprocess_config.json (from the model's inference.yml + PaddleX source) and
 * models/pipeline_max960.json (the max960 pipeline config and PaddleX defaults for null values).
 * Each JSON entry is {"value": ..., "source": ...}. A missing or mistyped value throws.
 */
data class PpOcrConfig(
    // Detection
    val detImgMode: String,
    val detMean: FloatArray,
    val detStd: FloatArray,
    val detScale: Float,
    val detNormalizeOrder: String,
    val detLimitSideLen: Int,
    val detLimitType: String,
    val detMaxSideLimit: Int,
    val dbThresh: Float,
    val dbBoxThresh: Float,
    val dbUnclipRatio: Float,
    val dbMaxCandidates: Int,
    val dbUseDilation: Boolean,
    val dbScoreMode: String,
    val dbBoxType: String,
    val dbMinSize: Int,
    // Recognition
    val recImgMode: String,
    val recImageShape: IntArray,   // C, H, W_min
    val recMaxWidth: Int,
    val recNormScale: Float,
    val recNormMean: Float,
    val recNormStd: Float,
    val recBatchSize: Int,
    val recScoreThresh: Float,
    val ctcClasses: Int,
    val ctcBlankIndex: Int,
    val spaceIndex: Int,
) {
    companion object {
        fun parse(preprocessJson: String, pipelineJson: String): PpOcrConfig {
            val pre = JSONObject(preprocessJson)
            val det = pre.getJSONObject("det")
            val rec = pre.getJSONObject("rec")
            val pipe = JSONObject(pipelineJson)

            val cfg = PpOcrConfig(
                detImgMode = det.str("img_mode"),
                detMean = det.floats("normalize_mean"),
                detStd = det.floats("normalize_std"),
                detScale = det.float("normalize_scale"),
                detNormalizeOrder = det.str("normalize_order"),
                detLimitSideLen = pipe.int("det_limit_side_len"),
                detLimitType = pipe.str("det_limit_type"),
                detMaxSideLimit = pipe.int("det_max_side_limit"),
                dbThresh = det.float("db_thresh"),
                dbBoxThresh = det.float("db_box_thresh"),
                dbUnclipRatio = det.float("db_unclip_ratio"),
                dbMaxCandidates = det.int("db_max_candidates"),
                dbUseDilation = det.optOrDefault("db_use_dilation", pipe) { o, k -> o.value(k) as Boolean },
                dbScoreMode = det.optOrDefault("db_score_mode", pipe) { o, k -> o.str(k) },
                dbBoxType = det.optOrDefault("db_box_type", pipe) { o, k -> o.str(k) },
                dbMinSize = pipe.int("db_min_size"),
                recImgMode = rec.str("img_mode"),
                recImageShape = rec.ints("rec_image_shape"),
                recMaxWidth = rec.int("rec_max_width"),
                recNormScale = pipe.float("rec_norm_scale"),
                recNormMean = pipe.float("rec_norm_mean"),
                recNormStd = pipe.float("rec_norm_std"),
                recBatchSize = pipe.int("rec_batch_size"),
                recScoreThresh = pipe.float("rec_score_thresh"),
                ctcClasses = rec.int("ctc_classes"),
                ctcBlankIndex = rec.ints("ctc_blank_index").single(),
                spaceIndex = rec.int("space_appended_at_index"),
            )
            cfg.validate()
            return cfg
        }

        private fun JSONObject.value(key: String): Any {
            val entry = optJSONObject(key) ?: throw ConfigException("missing \"$key\"")
            if (!entry.has("value") || entry.isNull("value")) throw ConfigException("\"$key\" has no value")
            return entry.get("value")
        }

        private fun JSONObject.str(key: String) = value(key) as? String ?: throw ConfigException("\"$key\" is not a string")
        private fun JSONObject.num(key: String) = value(key) as? Number ?: throw ConfigException("\"$key\" is not a number")
        private fun JSONObject.int(key: String) = num(key).toInt()
        private fun JSONObject.float(key: String) = num(key).toFloat()
        private fun JSONObject.array(key: String) =
            value(key) as? org.json.JSONArray ?: throw ConfigException("\"$key\" is not an array")
        private fun JSONObject.floats(key: String) = array(key).let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }
        private fun JSONObject.ints(key: String) = array(key).let { a -> IntArray(a.length()) { a.getInt(it) } }

        /** Uses the model config's value when set; when it is explicitly null, the PaddleX default in [pipe]. */
        private fun <T> JSONObject.optOrDefault(key: String, pipe: JSONObject, read: (JSONObject, String) -> T): T {
            val entry = optJSONObject(key) ?: throw ConfigException("missing \"$key\"")
            return if (entry.isNull("value")) read(pipe, key) else read(this, key)
        }
    }

    private fun validate() {
        fun check(ok: Boolean, what: String) { if (!ok) throw ConfigException(what) }
        check(detImgMode == "BGR" && recImgMode == "BGR", "expected BGR input, got det=$detImgMode rec=$recImgMode")
        check(detMean.size == 3 && detStd.size == 3, "det mean/std must have 3 channels")
        check(detNormalizeOrder == "hwc", "unsupported det normalize order $detNormalizeOrder")
        check(detLimitType == "max", "only limit_type=max is implemented, got $detLimitType")
        check(dbScoreMode == "fast", "only score_mode=fast is implemented, got $dbScoreMode")
        check(dbBoxType == "quad", "only box_type=quad is implemented, got $dbBoxType")
        check(!dbUseDilation, "use_dilation=true is not implemented")
        check(recImageShape.size == 3 && recImageShape[0] == 3, "rec_image_shape must be [3, H, W]")
        check(ctcBlankIndex == 0 && spaceIndex == ctcClasses - 1, "unexpected CTC layout")
    }
}

class ConfigException(message: String) : IllegalStateException("PP-OCR config: $message")
