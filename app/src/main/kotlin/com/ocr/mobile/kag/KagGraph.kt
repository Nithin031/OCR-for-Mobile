package com.ocr.mobile.kag

import org.json.JSONArray
import org.json.JSONObject

/**
 * The scheme knowledge graph (data/seed/graph.json of the web app), in memory.
 * Mirrors MemoryGraph in backend/app/graph/store.py: a scheme is expanded with its department,
 * portal, states, rules, required documents and the documents that describe it (DESCRIBED_IN).
 */
class KagGraph(json: String, private val documents: List<KagDocument>) {

    data class LifeEvent(val code: String, val name: String, val keywords: Map<String, List<String>>)
    data class Rule(val code: String, val text: String, val supportedBy: String?)
    data class Named(val code: String, val name: String, val url: String? = null)

    data class Scheme(
        val code: String,
        val name: String,
        val shortName: String,
        val summary: String,
        val benefit: String,
        val lifeEvents: List<String>,
        val department: Named?,
        val portal: Named?,
        val states: List<String>,
        val rules: List<Rule>,
        val documents: List<Named>,
        val sourceDocs: List<String>,
    )

    val lifeEvents: List<LifeEvent>
    val schemes: List<Scheme>

    init {
        val root = JSONObject(json)
        fun index(key: String): Map<String, JSONObject> =
            root.optJSONArray(key).objects().associateBy { it.getString("code") }
        val depts = index("departments")
        val portals = index("portals")
        val docReqs = index("document_requirements")
        val docIds = documents.map { it.id }.toSet()

        lifeEvents = root.optJSONArray("life_events").objects().map { le ->
            val kw = le.optJSONObject("keywords")
            LifeEvent(
                code = le.getString("code"),
                name = le.getString("name"),
                keywords = kw?.keys()?.asSequence()?.associateWith { kw.getJSONArray(it).strings() } ?: emptyMap(),
            )
        }
        schemes = root.optJSONArray("schemes").objects().map { s ->
            val code = s.getString("code")
            val sourceDoc = s.optString("source_doc").ifEmpty { null }
            // {source_doc} (when that document exists) ∪ documents tagged with this scheme, sorted.
            val sourceDocs = (listOfNotNull(sourceDoc?.takeIf { it in docIds }) +
                documents.filter { code in it.schemeCodes }.map { it.id }).toSortedSet().toList()
            Scheme(
                code = code,
                name = s.getString("name"),
                shortName = s.optString("short_name").ifEmpty { s.getString("name") },
                summary = s.optString("summary"),
                benefit = s.optString("benefit"),
                lifeEvents = s.optJSONArray("life_events").strings(),
                department = depts[s.optString("department")]?.let { Named(it.getString("code"), it.getString("name")) },
                portal = portals[s.optString("portal")]?.let { Named(it.getString("code"), it.getString("name"), it.optString("url").ifEmpty { null }) },
                states = s.optJSONArray("states").strings(),
                rules = s.optJSONArray("rules").objects().map {
                    Rule(it.getString("code"), it.getString("text"), it.optString("supported_by").ifEmpty { null })
                },
                documents = s.optJSONArray("documents").strings().mapNotNull { c ->
                    docReqs[c]?.let { Named(it.getString("code"), it.getString("name")) }
                },
                sourceDocs = sourceDocs,
            )
        }
    }

    fun scheme(code: String): Scheme? = schemes.firstOrNull { it.code == code }

    fun schemesForLifeEvent(code: String, state: String? = null): List<Scheme> = schemes.filter { s ->
        code in s.lifeEvents && (state == null || s.states.isEmpty() || setOf(state, "IN").any { it in s.states })
    }

    /** backend/app/ingestion/pipeline.py detect_scheme_mentions. */
    fun detectSchemeMentions(text: String): List<String> {
        val lower = text.lowercase()
        return schemes.filter { s ->
            val aliases = mutableSetOf(s.name, s.shortName, s.code.replace("_", " "))
            Regex("""\(([^)]+)\)""").find(s.name)?.let { aliases += it.groupValues[1] }
            aliases += EXTRA_ALIASES[s.code].orEmpty()
            aliases.any { a -> a.length > 2 && Regex("""\b""" + Regex.escape(a.lowercase()) + """\b""").containsMatchIn(lower) }
        }.map { it.code }
    }

    companion object {
        private val EXTRA_ALIASES = mapOf(
            "PM_KISAN" to setOf("pm-kisan", "pm kisan"),
            "PMFBY" to setOf("pmfby", "fasal bima", "crop insurance"),
            "KCC_CALAMITY_RELIEF" to setOf("kisan credit card", "kcc", "crop loan"),
            "CROP_LOSS_RELIEF_KA" to setOf("input subsidy", "sdrf", "ndrf", "crop loss relief"),
        )

        private fun JSONArray?.objects(): List<JSONObject> =
            if (this == null) emptyList() else (0 until length()).map { getJSONObject(it) }

        private fun JSONArray?.strings(): List<String> =
            if (this == null) emptyList() else (0 until length()).map { getString(it) }
    }
}
