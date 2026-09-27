package com.ocr.mobile.kag

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.util.BitSet

class KagRetrieverTest {

    private val graphJson = """
    {
      "life_events": [{"code": "CROP_DAMAGE", "name": "Crop Damage",
        "keywords": {"en": ["crop", "rain", "flood"], "kn": ["ಬೆಳೆ", "ಮಳೆ"]}}],
      "departments": [{"code": "MOAFW", "name": "Ministry of Agriculture"}],
      "states": [{"code": "IN", "name": "India"}],
      "portals": [{"code": "PMFBY_P", "name": "PMFBY Portal", "url": "https://pmfby.gov.in/"}],
      "document_requirements": [{"code": "LAND_RECORD", "name": "Land record (RTC)"}],
      "schemes": [
        {"code": "PMFBY", "name": "Pradhan Mantri Fasal Bima Yojana (PMFBY)", "short_name": "Crop Insurance",
         "summary": "Crop insurance.", "benefit": "Claim for crop loss.", "life_events": ["CROP_DAMAGE"],
         "department": "MOAFW", "states": ["IN"], "portal": "PMFBY_P", "documents": ["LAND_RECORD"],
         "source_doc": "missing-doc", "rules": [{"code": "PMFBY_R1", "text": "Must be insured.", "supported_by": "d1"}]},
        {"code": "PM_KISAN", "name": "PM-Kisan Samman Nidhi", "short_name": "PM-Kisan", "summary": "Income support.",
         "benefit": "Rs 6000 per year.", "life_events": [], "department": "MOAFW", "states": ["IN"], "documents": [], "rules": []}
      ]
    }
    """.trimIndent()

    private val docs = listOf(
        KagDocument("d1", "PMFBY guidelines", "pmfby.gov.in", "https://pmfby.gov.in/", listOf("PMFBY")),
        KagDocument("d2", "PM-Kisan FAQ", "pmkisan.gov.in", null, listOf("PM_KISAN")),
        KagDocument("d3", "Bank accounts", "sbi.co.in", null, emptyList()),
    )
    private val chunks = listOf(
        KagChunk("d1:0", "d1", 0, null, "Farmers must intimate crop loss due to hailstorm within 72 hours.", listOf("PMFBY")),
        KagChunk("d1:1", "d1", 1, null, "Premium for kharif crops is 2 percent.", listOf("PMFBY")),
        KagChunk("d2:0", "d2", 0, null, "PM-Kisan gives income support to farmers.", listOf("PM_KISAN")),
        KagChunk("d3:0", "d3", 0, null, "A savings bank account needs KYC documents.", emptyList()),
    )

    private fun retriever(): KagRetriever {
        val postings = HashMap<String, BitSet>()
        chunks.forEachIndexed { i, c ->
            Regex("""\w+""").findAll(c.content.lowercase()).forEach { postings.getOrPut(it.value) { BitSet() }.set(i) }
        }
        return KagRetriever(KagGraph(graphJson, docs), docs.associateBy { it.id }, chunks, postings)
    }

    @Test
    fun tokensDropStopwordsAndKeepIndicLettersWithMarks() {
        assertEquals(listOf("documents", "pmfby"), KagRetriever.tokens("What are the documents for PMFBY?"))
        // Python's \w splits at vowel signs (combining marks), so "ಬೆಳೆ" gives single-letter pieces that are dropped.
        assertTrue(KagRetriever.tokens("ಬೆಳೆ").none { it.length > 1 && it.any { c -> Character.getType(c) == Character.NON_SPACING_MARK.toInt() } })
    }

    @Test
    fun rareWordsOutweighCommonOnes() {
        val r = retriever()
        val top = r.keywordSearch("hailstorm crop").first()
        assertEquals(0, top.first) // the only chunk with "hailstorm"
    }

    @Test
    fun lifeEventNeedsTwoSignals() {
        val g = KagGraph(graphJson, docs)
        assertNull(KagQueryUnderstanding.detectLifeEvent(g, "my crop"))
        assertEquals("CROP_DAMAGE", KagQueryUnderstanding.detectLifeEvent(g, "rain destroyed my crop")?.code)
    }

    @Test
    fun understandsIntentAndSchemes() {
        val r = retriever()
        val q = r.understand("What documents are needed for crop insurance?")
        assertEquals("documents", q.intent)
        assertEquals(listOf("PMFBY"), q.schemeCodes)
        val followUp = r.understand("How do I apply?", contextSchemes = listOf("PM_KISAN"))
        assertEquals("how_to_apply", followUp.intent)
        assertTrue(followUp.fromContext)
        assertTrue(followUp.retrievalQuery.endsWith("PM-Kisan"))
    }

    @Test
    fun retrieveReturnsGraphFactsAnchorsAndBoostedChunks() {
        val r = retriever()
        val result = r.retrieve(r.understand("How do I claim PMFBY after hailstorm crop loss?"))
        assertEquals(listOf("PMFBY"), result.schemes.map { it.code })
        assertEquals(listOf("fact_pmfby", "fact_pmfby_r1", "fact_pmfby_docs"), result.facts.map { it.id })
        assertEquals("PMFBY guidelines", result.facts.first().supportingDocument) // source_doc missing -> tagged doc d1
        assertEquals("d1:0", result.chunks.first().id)
        assertTrue("graph-linked" in result.chunks.first().retrieval)
        assertEquals("d1:0", result.anchors.getValue("PMFBY").id)
    }

    @Test
    fun contextRespectsCharacterBudget() {
        val r = retriever()
        val result = r.retrieve(r.understand("PMFBY hailstorm premium documents"))
        val (context, used) = KagRetriever.buildContext(result, maxChars = 400)
        assertTrue(context.length <= 400 + 40)
        assertTrue(used.isNotEmpty())
        assertTrue(used.all { "[${it.id}]" in context })
    }

    @Test
    fun documentsQuestionPutsRequiredDocumentsFactFirst() {
        val r = retriever()
        val result = r.retrieve(r.understand("What documents do I need for crop insurance PMFBY"))
        assertEquals("documents", result.query.intent)
        val (context, used) = KagRetriever.buildContext(result, maxChars = 250)
        assertEquals("fact_pmfby_docs", used.first().id)
        assertTrue("Land record (RTC)" in context)
    }

    @Test
    fun languageDetectionByScript() {
        assertEquals("kn", KagQueryUnderstanding.detectLanguage("ಬೆಳೆ ಹಾನಿ"))
        assertEquals("hi", KagQueryUnderstanding.detectLanguage("फसल नुकसान"))
        assertEquals("en", KagQueryUnderstanding.detectLanguage("crop loss"))
    }
}
