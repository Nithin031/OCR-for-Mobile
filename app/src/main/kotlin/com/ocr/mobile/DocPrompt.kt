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

    data class Built(val prompt: String, val linesUsed: Int, val linesTotal: Int)

    fun build(lines: List<String>, question: String): Built {
        val used = StringBuilder()
        var count = 0
        for (line in lines) {
            if (used.length + line.length + 1 > MAX_DOC_CHARS) break
            used.append(line).append('\n')
            count++
        }
        val note = if (count < lines.size) "(${lines.size - count} more lines were left out because of length.)\n" else ""
        val prompt = "$INSTRUCTIONS\n\nDocument text:\n$used$note\nQuestion: ${question.trim()}"
        return Built(prompt, count, lines.size)
    }
}
