package com.ocr.mobile.kag

import java.util.BitSet
import kotlin.math.ln

data class KagDocument(val id: String, val title: String, val publisher: String?, val url: String?, val schemeCodes: List<String>)

data class KagChunk(
    val id: String,
    val documentId: String,
    val chunkIndex: Int,
    val section: String?,
    val content: String,
    val schemeCodes: List<String>,
)

/** One piece of evidence given to the LLM: a graph fact or a document chunk. */
data class KagEvidence(
    val id: String,
    val type: String,               // "graph_fact" | "chunk"
    val text: String,
    val score: Double,
    val retrieval: List<String>,
    val title: String,
    val publisher: String?,
    val url: String?,
    val section: String?,
    val schemeCodes: List<String> = emptyList(),
    val supportingDocument: String? = null,
    val relevant: Boolean = true,
)

data class KagResult(
    val query: KagQuery,
    val schemes: List<KagGraph.Scheme>,
    val facts: List<KagEvidence>,
    val chunks: List<KagEvidence>,
    val anchors: Map<String, KagEvidence>,
) {
    fun hasRelevantEvidence(): Boolean = facts.isNotEmpty() || chunks.any { it.relevant }
}

/**
 * Offline mirror of backend/app/kag/retriever.py. Differences, all forced by running on a phone:
 *  - no vector search (no embedding model on the device): rank fusion runs over the keyword ranking only;
 *  - keyword search scores every matching chunk (the server caps Postgres candidates at 300, unordered);
 *  - lexemes are precomputed by tools/kag/build_kag_db.py instead of Postgres to_tsvector('simple').
 */
class KagRetriever(
    val graph: KagGraph,
    private val documents: Map<String, KagDocument>,
    private val chunks: List<KagChunk>,
    private val postings: Map<String, BitSet>,
) {

    fun understand(question: String, contextSchemes: List<String> = emptyList()) =
        KagQueryUnderstanding.understand(graph, question, contextSchemes)

    fun retrieve(q: KagQuery, topK: Int = 8, extraQuery: String = ""): KagResult {
        var schemes = graphRetrieve(q)
        val rq = "${q.retrievalQuery} $extraQuery".trim()
        val kw = keywordSearch(rq)

        val fused = LinkedHashMap<Int, Double>()
        val methods = HashMap<Int, MutableList<String>>()
        kw.forEachIndexed { rank, (ci, _) ->
            fused[ci] = (fused[ci] ?: 0.0) + 1.0 / (RRF_K + rank)
            methods.getOrPut(ci) { mutableListOf() } += "keyword"
        }

        val graphCodes = schemes.map { it.code }.toSet()
        val candidates = fused.map { (ci, base) ->
            val chunk = chunks[ci]
            var score = base
            // Graph-guided boost: chunks describing schemes found in the graph.
            if (graphCodes.isNotEmpty() && chunk.schemeCodes.any { it in graphCodes }) {
                score *= 1.08
                methods.getValue(ci) += "graph-linked"
            }
            // Intent-aware boost on section headings.
            val sec = chunk.section.orEmpty().lowercase()
            if (INTENT_SECTIONS[q.intent].orEmpty().any { it in sec }) score *= 1.05
            chunkEvidence(chunk, score, methods.getValue(ci), relevant = "keyword" in methods.getValue(ci))
        }.sortedByDescending { it.score }

        val perDoc = HashMap<String, Int>()
        val picked = ArrayList<KagEvidence>()
        for (e in candidates) {
            val d = docOf(e)
            if ((perDoc[d] ?: 0) >= MAX_PER_DOC) continue
            perDoc[d] = (perDoc[d] ?: 0) + 1
            picked += e
            if (picked.size >= topK) break
        }

        // Evidence -> graph expansion: if the graph query found nothing, use the schemes of the top evidence.
        if (schemes.isEmpty()) {
            val codes = mutableListOf<String>()
            for (e in picked.take(3)) if (e.relevant) codes += e.schemeCodes.filter { it !in codes }
            schemes = codes.take(2).mapNotNull { graph.scheme(it) }
        }
        return KagResult(q, schemes, graphFacts(schemes), picked, anchorChunks(schemes, q.intent))
    }

    private fun docOf(e: KagEvidence) = e.id.substringBeforeLast(':')

    private fun graphRetrieve(q: KagQuery): List<KagGraph.Scheme> {
        val out = LinkedHashMap<String, KagGraph.Scheme>()
        q.lifeEvent?.let { le -> graph.schemesForLifeEvent(le.code).forEach { out[it.code] = it } }
        for (code in q.schemeCodes) graph.scheme(code)?.let { out[code] = it }
        return out.values.toList()
    }

    /** IDF-rescored keyword search, like keyword_search() on the server. Returns (chunk index, score). */
    fun keywordSearch(query: String, limit: Int = 20): List<Pair<Int, Double>> {
        val toks = tokens(query).map { t -> t.filter(::isWordChar) }.filter { it.isNotEmpty() }.distinct()
        if (toks.isEmpty()) return emptyList()
        val total = chunks.size.coerceAtLeast(1)
        val idf = toks.associateWith { t -> ln(1.0 + total.toDouble() / (1 + (postings[t]?.cardinality() ?: 0))) }
        val any = BitSet()
        toks.forEach { t -> postings[t]?.let { any.or(it) } }
        val scored = ArrayList<Pair<Int, Double>>()
        var ci = any.nextSetBit(0)
        while (ci >= 0) {
            val sc = toks.sumOf { t -> if (postings[t]?.get(ci) == true) idf.getValue(t) else 0.0 }
            if (sc > 0) scored += ci to sc
            ci = any.nextSetBit(ci + 1)
        }
        scored.sortByDescending { it.second }
        return scored.take(limit)
    }

    private fun graphFacts(schemes: List<KagGraph.Scheme>): List<KagEvidence> {
        val out = ArrayList<KagEvidence>()
        for (s in schemes) {
            val dept = s.department?.name.orEmpty()
            val portal = s.portal
            val supporting = s.sourceDocs.firstOrNull()?.let { documents[it]?.title ?: it }
            fun fact(id: String, text: String, section: String, sup: String?) = KagEvidence(
                id = factId(id), type = "graph_fact", text = text, score = 0.0, retrieval = listOf("graph"),
                title = "Knowledge Graph", publisher = dept, url = portal?.url, section = section,
                schemeCodes = listOf(s.code), supportingDocument = sup,
            )
            out += fact(s.code, "Scheme: ${s.name}. ${s.summary} Benefit: ${s.benefit} " +
                "Managed by: $dept. Apply at: ${portal?.name ?: "n/a"}.", "Scheme", supporting)
            for (r in s.rules) {
                out += fact(r.code, "Eligibility rule for ${s.name}: ${r.text}", "Eligibility rule",
                    r.supportedBy?.let { documents[it]?.title ?: it })
            }
            if (s.documents.isNotEmpty()) {
                out += fact(s.code + "_docs", "Required documents for ${s.name}: " +
                    s.documents.joinToString("; ") { it.name } + ".", "Required documents", supporting)
            }
        }
        return out
    }

    /** Graph -> Document -> Chunk: the best chunk of a document linked to each scheme. */
    private fun anchorChunks(schemes: List<KagGraph.Scheme>, intent: String): Map<String, KagEvidence> {
        val prefs = ANCHOR_SECTIONS[intent].orEmpty() + listOf("overview", "benefit", "assistance amount", "kisan credit card")
        val out = LinkedHashMap<String, KagEvidence>()
        for (s in schemes) {
            if (s.sourceDocs.isEmpty()) continue
            val docIds = s.sourceDocs.toSet()
            val rows = chunks.filter { it.documentId in docIds }.sortedBy { it.chunkIndex }
            if (rows.isEmpty()) continue
            fun rank(c: KagChunk): Int {
                val sec = c.section.orEmpty().lowercase()
                prefs.forEachIndexed { i, p -> if (p in sec) return i }
                return prefs.size + c.chunkIndex
            }
            val best = rows.minBy(::rank)
            out[s.code] = chunkEvidence(best, 0.0, listOf("graph-anchor"), relevant = true)
        }
        return out
    }

    private fun chunkEvidence(c: KagChunk, score: Double, methods: List<String>, relevant: Boolean): KagEvidence {
        val doc = documents[c.documentId]
        return KagEvidence(
            id = c.id, type = "chunk", text = c.content, score = score, retrieval = methods,
            title = doc?.title ?: c.documentId, publisher = doc?.publisher, url = doc?.url, section = c.section,
            schemeCodes = c.schemeCodes, relevant = relevant,
        )
    }

    companion object {
        const val RRF_K = 5
        const val MAX_PER_DOC = 3

        private val STOPWORDS = ("a an the is are was were be been i me my we our you your it its of in on at to for from with and or " +
            "not no do does did can could what which who how when where why this that these there here get got have has had will " +
            "would should shall may might about into over under by as if so than then them they he she his her help please tell " +
            "want need know").split(" ").toSet()

        val INTENT_SECTIONS = mapOf(
            "eligibility" to listOf("eligib", "exclusion", "condition"),
            "documents" to listOf("document", "required"),
            "how_to_apply" to listOf("apply", "registration", "intimate", "what the farmer should do", "how to"),
            "amount" to listOf("amount", "benefit", "premium", "overview"),
        )
        private val ANCHOR_SECTIONS = mapOf(
            "documents" to listOf("document"), "eligibility" to listOf("eligib"),
            "how_to_apply" to listOf("apply", "intimate", "should do", "registration"),
            "amount" to listOf("amount", "benefit", "premium", "claim settlement"),
        )

        /** Python's \w: letters, numbers and underscore (not combining marks). */
        fun isWordChar(c: Char): Boolean = c == '_' || Character.isLetter(c) || when (Character.getType(c).toByte()) {
            Character.DECIMAL_DIGIT_NUMBER, Character.LETTER_NUMBER, Character.OTHER_NUMBER -> true
            else -> false
        }

        /** _tokens(): runs of [\w-], lower-cased, without stopwords and 1-char tokens, de-duplicated, max 24. */
        fun tokens(q: String): List<String> {
            val out = LinkedHashSet<String>()
            val cur = StringBuilder()
            fun flush() {
                val t = cur.toString()
                cur.clear()
                if (t.length > 1 && t !in STOPWORDS) out += t
            }
            for (ch in q.lowercase()) if (isWordChar(ch) || ch == '-') cur.append(ch) else flush()
            flush()
            return out.take(24)
        }

        fun factId(code: String) = "fact_" + code.lowercase().replace(Regex("[^a-z0-9]+"), "_").trim('_')

        /** backend/app/kag/retriever.py build_context, trimmed to [maxChars] for the on-device model's context. */
        fun buildContext(r: KagResult, maxChars: Int, maxChunkChars: Int = 450): Pair<String, List<KagEvidence>> {
            val used = ArrayList<KagEvidence>()
            val sb = StringBuilder("GRAPH FACTS (knowledge graph):\n")
            fun fits(s: String) = sb.length + s.length <= maxChars
            if (r.facts.isEmpty()) sb.append("(none)\n")
            for (f in r.facts) {
                val sup = f.supportingDocument?.let { " (supported by: $it)" }.orEmpty()
                val line = "[${f.id}] ${f.text}$sup\n"
                if (!fits(line)) break
                sb.append(line); used += f
            }
            sb.append("\nRETRIEVED DOCUMENT EVIDENCE:\n")
            val chunkIds = r.chunks.map { it.id }.toSet()
            val all = r.chunks + r.anchors.values.filter { it.id !in chunkIds }
            var any = false
            for (c in all) {
                val block = "[${c.id}] Publisher: ${c.publisher} | Document: ${c.title}\n${c.text.take(maxChunkChars)}\n\n"
                if (!fits(block)) continue
                sb.append(block); used += c; any = true
            }
            if (!any) sb.append("(none)\n")
            return sb.toString().trimEnd() to used
        }
    }
}
