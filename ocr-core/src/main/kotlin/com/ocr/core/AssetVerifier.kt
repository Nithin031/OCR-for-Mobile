package com.ocr.core

import android.content.Context
import android.util.Log
import org.json.JSONObject
import java.security.MessageDigest

object AssetVerifier {

    private const val TAG = "AssetVerifier"
    private const val MANIFEST_PATH = "models/manifest.json"

    data class VerificationResult(
        val okFiles: List<String>,
        val missingFiles: List<String>,
        // path → "expected <hex> / got <hex>"
        val corruptedFiles: Map<String, String>,
    )

    fun verify(context: Context): VerificationResult {
        val assets = context.assets

        val manifestText = runCatching {
            assets.open(MANIFEST_PATH).bufferedReader().readText()
        }.getOrElse {
            Log.e(TAG, "Cannot open $MANIFEST_PATH — was the Gradle copyModels task run?", it)
            return VerificationResult(emptyList(), listOf(MANIFEST_PATH), emptyMap())
        }

        val files = JSONObject(manifestText).getJSONObject("files")

        val ok = mutableListOf<String>()
        val missing = mutableListOf<String>()
        val corrupted = mutableMapOf<String, String>()

        files.keys().forEach { relPath ->
            val entry = files.getJSONObject(relPath)
            val expectedSha = entry.getString("sha256")
            val assetPath = "models/$relPath"

            val bytes = runCatching { assets.open(assetPath).readBytes() }.getOrNull()
            if (bytes == null) {
                missing.add(assetPath)
                return@forEach
            }

            val actualSha = sha256Hex(bytes)
            if (actualSha == expectedSha) {
                ok.add(assetPath)
            } else {
                corrupted[assetPath] = "expected $expectedSha / got $actualSha"
            }
        }

        return VerificationResult(ok, missing, corrupted)
    }

    private fun sha256Hex(bytes: ByteArray): String {
        val digest = MessageDigest.getInstance("SHA-256").digest(bytes)
        return digest.joinToString("") { "%02x".format(it) }
    }
}
