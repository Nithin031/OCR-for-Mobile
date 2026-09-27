package com.ocr.mobile.kag

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.util.Log
import org.json.JSONArray
import java.io.File
import java.util.BitSet

/**
 * The offline KAG database: assets/kag/kag.db (built by tools/kag/build_kag_db.py from the
 * BuildForBillions26 web app) is copied to app storage once, opened read-only and loaded into memory.
 */
object KagStore {

    private const val ASSET = "kag/kag.db"
    private const val TAG = "KagStore"

    data class Info(val documents: Int, val chunks: Int, val schemes: Int, val sourceCommit: String, val loadMs: Long)

    fun available(context: Context): Boolean =
        runCatching { context.assets.list("kag")?.contains("kag.db") == true }.getOrDefault(false)

    /** Must be called off the main thread. */
    fun load(context: Context): Pair<KagRetriever, Info> {
        val start = System.currentTimeMillis()
        val file = ensureCopied(context)
        val db = SQLiteDatabase.openDatabase(file.path, null, SQLiteDatabase.OPEN_READONLY)
        try {
            val meta = HashMap<String, String>()
            db.rawQuery("SELECT key, value FROM meta", null).use { c -> while (c.moveToNext()) meta[c.getString(0)] = c.getString(1) }
            check(meta["schema_version"] == "1") { "kag.db schema ${meta["schema_version"]}, expected 1" }

            val documents = LinkedHashMap<String, KagDocument>()
            db.rawQuery("SELECT id, title, publisher, url, scheme_codes FROM documents", null).use { c ->
                while (c.moveToNext()) {
                    val d = KagDocument(c.getString(0), c.getString(1), c.getStringOrNull(2), c.getStringOrNull(3), codes(c.getString(4)))
                    documents[d.id] = d
                }
            }
            val graphJson = db.rawQuery("SELECT json FROM graph WHERE id = 1", null).use { c ->
                check(c.moveToFirst()) { "kag.db has no graph" }
                c.getString(0)
            }

            val chunks = ArrayList<KagChunk>()
            val postings = HashMap<String, BitSet>()
            db.rawQuery(
                "SELECT id, document_id, chunk_index, section, content, scheme_codes, lexemes FROM chunks ORDER BY document_id, chunk_index",
                null,
            ).use { c ->
                while (c.moveToNext()) {
                    val index = chunks.size
                    chunks += KagChunk(c.getString(0), c.getString(1), c.getInt(2), c.getStringOrNull(3), c.getString(4), codes(c.getString(5)))
                    for (lex in c.getString(6).split(' ')) if (lex.isNotEmpty()) postings.getOrPut(lex) { BitSet() }.set(index)
                }
            }
            check(chunks.size.toString() == meta["chunks"]) { "kag.db chunk count ${chunks.size} != meta ${meta["chunks"]}" }

            val graph = KagGraph(graphJson, documents.values.toList())
            val info = Info(documents.size, chunks.size, graph.schemes.size, meta["source_commit"].orEmpty(), System.currentTimeMillis() - start)
            Log.i(TAG, "Loaded ${info.documents} documents, ${info.chunks} chunks, ${info.schemes} schemes in ${info.loadMs} ms")
            return KagRetriever(graph, documents, chunks, postings) to info
        } finally {
            db.close()
        }
    }

    /** Copies the asset when missing or when the APK carries a different size (a rebuilt database). */
    private fun ensureCopied(context: Context): File {
        val target = context.getDatabasePath("kag.db")
        val assetSize = context.assets.openFd(ASSET).use { it.length }
        if (target.isFile && target.length() == assetSize) return target
        target.parentFile?.mkdirs()
        val part = File(target.path + ".part")
        context.assets.open(ASSET).use { input -> part.outputStream().use { input.copyTo(it) } }
        check(part.renameTo(target)) { "could not install kag.db" }
        return target
    }

    private fun codes(json: String): List<String> = JSONArray(json).let { a -> (0 until a.length()).map { a.getString(it) } }

    private fun android.database.Cursor.getStringOrNull(i: Int): String? = if (isNull(i)) null else getString(i)
}
