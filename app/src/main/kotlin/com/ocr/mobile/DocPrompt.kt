package com.ocr.mobile

/** Builds the assistant prompt from the OCR lines. The OCR text is never changed, only cut to fit. */
object DocPrompt {

    /** Keeps prompt + answer inside GemmaEngine.MAX_TOKENS (about 3 characters per token for OCR text). */
    const val MAX_DOC_CHARS = 2000

    const val SUMMARIZE = "Summarize this document in a few bullet points."
    const val ASKS_FOR = "What information does this form ask for?"

    private const val INSTRUCTIONS =
        "You help a person understand a scanned document. Its text was read by OCR and may contain " +
            "reading errors. Use only the document text below. If the answer is not in the text, say that " +
            "the document does not say. Never guess or correct names, ID numbers, dates or amounts: copy " +
            "them exactly as written. Answer in English and keep it short."

    /** With knowledge-base evidence the document text gets less room, so the total stays inside MAX_TOKENS. */
    const val MAX_DOC_CHARS_WITH_KAG = 1000
    /** When the question itself names a scheme, the knowledge base is the main source: less document text. */
    const val MAX_DOC_CHARS_SCHEME_QUESTION = 400
    const val MAX_KAG_CHARS = 1500

    // Core rules of the web app's KAG system prompt (backend/app/kag/prompts.py), shortened for a 1B model.
    private const val KAG_INSTRUCTIONS =
        "You help a person understand a scanned government document and related government schemes. " +
            "Use only the document text and the knowledge base evidence below. Do not invent eligibility rules, " +
            "documents, amounts, deadlines, phone numbers or URLs. If the evidence does not answer the question, say " +
            "you cannot verify it from the available sources. After a fact from the knowledge base, add its ID in " +
            "square brackets, e.g. [fact_pmfby]; use only IDs shown below. Copy names, ID numbers, dates and amounts " +
            "exactly as written. Answer in English, short and simple."

    data class Built(val prompt: String, val linesUsed: Int, val linesTotal: Int)

    fun build(lines: List<String>, question: String): Built {
        val (used, count, note) = fit(lines, MAX_DOC_CHARS)
        val prompt = "$INSTRUCTIONS\n\nDocument text:\n$used$note\nQuestion: ${question.trim()}"
        return Built(prompt, count, lines.size)
    }

    /** Same as [build] plus the KAG context from KagRetriever.buildContext. */
    fun buildWithKnowledge(lines: List<String>, question: String, kagContext: String, schemeQuestion: Boolean = false): Built {
        val (used, count, note) = fit(lines, if (schemeQuestion) MAX_DOC_CHARS_SCHEME_QUESTION else MAX_DOC_CHARS_WITH_KAG)
        val prompt = "$KAG_INSTRUCTIONS\n\nDocument text:\n$used$note\n$kagContext\n\nQuestion: ${question.trim()}"
        return Built(prompt, count, lines.size)
    }

    private fun fit(lines: List<String>, maxChars: Int): Triple<String, Int, String> {
        val used = StringBuilder()
        var count = 0
        for (line in lines) {
            if (used.length + line.length + 1 > maxChars) break
            used.append(line).append('\n')
            count++
        }
        val note = if (count < lines.size) "(${lines.size - count} more lines were left out because of length.)\n" else ""
        return Triple(used.toString(), count, note)
    }
}
