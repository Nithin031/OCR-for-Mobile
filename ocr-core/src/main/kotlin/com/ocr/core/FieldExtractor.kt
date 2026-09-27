package com.ocr.core

enum class FieldKind(val label: String) {
    TEXT("Field"),
    DATE("Date"),
    PAN("PAN"),
    AADHAAR("Aadhaar"),
    PHONE("Phone"),
    EMAIL("Email"),
    IFSC("IFSC"),
    PINCODE("PIN code"),
    AMOUNT("Amount"),
}

/** A value found on the page. [value] is copied exactly as OCR read it; nothing is corrected. */
data class DocField(val kind: FieldKind, val label: String, val value: String, val row: Int)

/**
 * Finds form fields in OCR rows: "Label: value" pairs (in one line, or a "Label:" line followed by its
 * value in the same row) and well-known Indian formats (dates, PAN, Aadhaar, phone, e-mail, IFSC,
 * PIN code, amounts). A labelled value that matches a format takes that format's kind.
 */
object FieldExtractor {

    private class Pattern(val kind: FieldKind, val regex: Regex, val rowFilter: Regex? = null)

    private val months = "jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec"

    // Order matters: earlier patterns claim their text first, so e.g. an Aadhaar number is not also a phone.
    private val patterns = listOf(
        Pattern(FieldKind.EMAIL, Regex("""[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}""")),
        Pattern(FieldKind.PAN, Regex("""\b[A-Z]{5}[0-9]{4}[A-Z]\b""")),
        Pattern(FieldKind.IFSC, Regex("""\b[A-Z]{4}0[A-Z0-9]{6}\b""")),
        Pattern(FieldKind.AADHAAR, Regex("""(?<![\d+])[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)""")),
        Pattern(FieldKind.PHONE, Regex("""(?<![\d+])(?:\+91[ -]?|0)?[6-9]\d{4}[ -]?\d{5}(?!\d)""")),
        Pattern(
            FieldKind.DATE,
            Regex(
                """(?<!\d)(?:\d{1,2}[/.-]\d{1,2}[/.-](?:\d{4}|\d{2})|\d{4}-\d{2}-\d{2}|\d{1,2}(?:st|nd|rd|th)?[ -](?:$months)[a-z]*\.?,?[ -]\d{4})(?!\d)""",
                RegexOption.IGNORE_CASE,
            ),
        ),
        Pattern(FieldKind.AMOUNT, Regex("""(?:₹|\bRs\.?|\bINR)\s?\d[\d,]*(?:\.\d{1,2})?""", RegexOption.IGNORE_CASE)),
        Pattern(
            FieldKind.PINCODE,
            Regex("""(?<!\d)[1-9]\d{2} ?\d{3}(?!\d)"""),
            rowFilter = Regex("""\bpin\b|pin ?code|postal""", RegexOption.IGNORE_CASE),
        ),
    )

    private class Match(val kind: FieldKind, val value: String)
    private class Labeled(val label: String, val value: String)

    fun extract(rows: List<List<String>>): List<DocField> {
        val out = mutableListOf<DocField>()
        rows.forEachIndexed { index, items ->
            val matches = findPatterns(items.joinToString("   "))
            val labeled = findLabeled(items)
            val used = BooleanArray(matches.size)
            for (field in labeled) {
                val i = matches.indices.firstOrNull { !used[it] && field.value.contains(matches[it].value) }
                val kind = if (i != null) {
                    used[i] = true
                    matches[i].kind
                } else FieldKind.TEXT
                out += DocField(kind, field.label, field.value, index)
            }
            matches.forEachIndexed { i, m -> if (!used[i]) out += DocField(m.kind, m.kind.label, m.value, index) }
        }
        return out.distinctBy { Triple(it.kind, it.label.lowercase(), it.value) }
    }

    private fun findPatterns(text: String): List<Match> {
        val masked = StringBuilder(text)
        val found = mutableListOf<Pair<Int, Match>>()
        for (p in patterns) {
            if (p.rowFilter != null && !p.rowFilter.containsMatchIn(text)) continue
            for (m in p.regex.findAll(masked.toString())) {
                if (p.kind == FieldKind.DATE && !plausibleNumericDate(m.value)) continue
                found += m.range.first to Match(p.kind, text.substring(m.range))
                for (i in m.range) masked.setCharAt(i, ' ')
            }
        }
        return found.sortedBy { it.first }.map { it.second }
    }

    /** Rejects numeric "dates" such as 45/78/2020; month-name and ISO dates pass through. */
    private fun plausibleNumericDate(value: String): Boolean {
        val parts = value.split('/', '.', '-')
        if (parts.size != 3 || parts.any { p -> p.isEmpty() || !p.all(Char::isDigit) }) return true
        val (a, b) = parts[0].toInt() to parts[1].toInt()
        if (parts[0].length == 4) return b in 1..12 && parts[2].toInt() in 1..31
        return a in 1..31 && b in 1..31 && (a <= 12 || b <= 12)
    }

    private fun findLabeled(items: List<String>): List<Labeled> {
        val out = mutableListOf<Labeled>()
        var i = 0
        while (i < items.size) {
            val item = items[i].trim()
            val colon = item.indexOf(':')
            if (colon > 0) {
                val label = item.substring(0, colon).trim()
                val value = item.substring(colon + 1).trim()
                if (isLabel(label)) {
                    if (value.isNotEmpty() && !value.startsWith("//")) {
                        out += Labeled(label, value)
                    } else if (value.isEmpty() && i + 1 < items.size && !items[i + 1].contains(':')) {
                        out += Labeled(label, items[i + 1].trim())
                        i++
                    }
                }
            }
            i++
        }
        return out
    }

    /** A short phrase with at least one letter: "Name", "Date of Birth", "Mobile No." */
    private fun isLabel(label: String): Boolean =
        label.length in 1..MAX_LABEL_CHARS &&
            label.any(Char::isLetter) &&
            label.split(Regex("""\s+""")).size <= MAX_LABEL_WORDS

    private const val MAX_LABEL_CHARS = 40
    private const val MAX_LABEL_WORDS = 6
}
