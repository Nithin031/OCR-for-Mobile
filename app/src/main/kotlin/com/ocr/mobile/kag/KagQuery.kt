package com.ocr.mobile.kag

/**
 * Query understanding, mirroring backend/app/kag/query_understanding.py and kag/lang.py.
 * Offline there is no translation LLM, so non-English questions use the web app's glossary
 * fallback (the same path the server takes when no LLM is available).
 */
data class KagQuery(
    val question: String,
    val language: String,
    val retrievalQuery: String,
    val lifeEvent: KagGraph.LifeEvent?,
    val intent: String,
    val schemeCodes: List<String>,
    val fromContext: Boolean,
)

object KagQueryUnderstanding {

    private val GLOSSARY = mapOf(
        "आधार" to "aadhaar", "बैंक" to "bank account", "खाता" to "account", "दस्तावेज़" to "documents", "दस्तावेज" to "documents",
        "कागज" to "documents", "पात्र" to "eligible eligibility", "पात्रता" to "eligibility", "बीमा" to "insurance pmfby",
        "ऋण" to "loan", "कर्ज" to "loan", "आवेदन" to "apply application", "कितना" to "amount", "पैसा" to "amount money",
        "सर्वे" to "survey number", "ज़मीन" to "land record", "जमीन" to "land record", "किसान" to "farmer", "सम्मान निधि" to "pm-kisan",
        "ಆಧಾರ್" to "aadhaar", "ಬ್ಯಾಂಕ್" to "bank account", "ಖಾತೆ" to "account", "ದಾಖಲೆ" to "documents", "ದಾಖಲೆಗಳು" to "documents",
        "ಅರ್ಹತೆ" to "eligibility", "ವಿಮೆ" to "insurance pmfby", "ಸಾಲ" to "loan", "ಅರ್ಜಿ" to "apply application", "ಎಷ್ಟು" to "amount",
        "ಹಣ" to "amount money", "ಸರ್ವೆ" to "survey number", "ಪಹಣಿ" to "rtc pahani land record", "ಜಮೀನು" to "land record", "ರೈತ" to "farmer",
    )

    // Ordered like the Python dict: the first matching intent wins.
    private val INTENT_PATTERNS = linkedMapOf(
        "why" to Regex("""why did you|why do you say|where did you get|source of (this|that)|क्यों बताया|ಯಾಕೆ ಹೇಳಿದ"""),
        "documents" to Regex("""document|papers|proof|certificate|दस्तावेज|कागज|प्रमाण|ದಾಖಲೆ|ಪುರಾವೆ"""),
        "eligibility" to Regex("""eligib|qualify|can i get|am i|who can|पात्र|योग्य|ಅರ್ಹ"""),
        "how_to_apply" to Regex("""how (do|can|to) .*apply|apply|process|procedure|steps|register|आवेदन|कैसे|ಅರ್ಜಿ|ಹೇಗೆ"""),
        "amount" to Regex("""how much|amount|money|benefit|rupees|₹|कितना|राशि|ಎಷ್ಟು|ಮೊತ್ತ"""),
    )

    private val DISCOVER = Regex(
        "what help|any help|help me|can i get|what can i do|scheme|support|assistance|relief|compensation|" +
            "मदद|सहायता|योजना|राहत|ಸಹಾಯ|ನೆರವು|ಯೋಜನೆ|ಪರಿಹಾರ"
    )
    private val QUESTION_WORD = Regex("""\?|\b(how|when|what|which|where|why|who)\b""")

    fun detectLanguage(text: String, default: String = "en"): String {
        val deva = text.count { it in 'ऀ'..'ॿ' }
        val kann = text.count { it in 'ಀ'..'೿' }
        val latin = text.count { it in 'A'..'Z' || it in 'a'..'z' }
        return when {
            kann > maxOf(deva.toDouble(), latin * 0.3) -> "kn"
            deva > maxOf(kann.toDouble(), latin * 0.3) -> "hi"
            latin > 0 -> "en"
            else -> default
        }
    }

    fun detectLifeEvent(graph: KagGraph, text: String): KagGraph.LifeEvent? {
        val lower = text.lowercase()
        var best: KagGraph.LifeEvent? = null
        var bestHits = 0
        for (le in graph.lifeEvents) {
            var hits = 0
            for (words in le.keywords.values) for (w in words) {
                val found = if (w.any { it in 'a'..'z' || it in 'A'..'Z' }) {
                    Regex("""\b""" + Regex.escape(w.lowercase()) + """\b""").containsMatchIn(lower)
                } else {
                    w in text
                }
                if (found) hits++
            }
            if (hits > bestHits) {
                best = le
                bestHits = hits
            }
        }
        // At least 2 signals (e.g. "crop" + "rain") to avoid false positives.
        return if (bestHits >= 2) best else null
    }

    fun understand(graph: KagGraph, question: String, contextSchemes: List<String> = emptyList()): KagQuery {
        val lang = detectLanguage(question)
        var retrievalQuery = question
        if (lang != "en") {
            val extra = GLOSSARY.filterKeys { it in question }.values
            retrievalQuery = question + " " + extra.joinToString(" ")
        }
        val lifeEvent = detectLifeEvent(graph, "$question $retrievalQuery")
        if (lifeEvent != null && lang != "en") {
            retrievalQuery += " " + lifeEvent.name + " " + lifeEvent.keywords["en"].orEmpty().take(8).joinToString(" ")
        }
        val probe = "$question $retrievalQuery".lowercase()
        val broad = DISCOVER.containsMatchIn(probe) || !QUESTION_WORD.containsMatchIn(probe)
        var intent = if (lifeEvent != null && broad) "discover" else "general"
        for ((name, pattern) in INTENT_PATTERNS) {
            if (pattern.containsMatchIn(probe)) {
                if (!(lifeEvent != null && name in setOf("eligibility", "amount"))) intent = name
                break
            }
        }
        var schemes = graph.detectSchemeMentions(retrievalQuery).ifEmpty { graph.detectSchemeMentions(question) }
        var fromContext = false
        if (schemes.isEmpty() && contextSchemes.isNotEmpty()) {
            schemes = contextSchemes
            fromContext = true
            if (schemes.size <= 2 && lifeEvent == null) {
                val names = schemes.take(2).mapNotNull { graph.scheme(it)?.shortName }
                retrievalQuery = "$retrievalQuery ${names.joinToString(" ")}".trim()
            }
        }
        return KagQuery(question, lang, retrievalQuery, lifeEvent, intent, schemes, fromContext)
    }
}
