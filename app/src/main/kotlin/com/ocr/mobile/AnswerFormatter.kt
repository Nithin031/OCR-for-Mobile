package com.ocr.mobile

/**
 * Turns the model's raw Markdown-ish answer into clean blocks for display:
 * headings, bullets and numbered steps become blocks, **bold** becomes a styled run, stray
 * Markdown symbols are removed, and evidence IDs such as [fact_pmfby] or [sf-3fa…:4] become
 * short reference numbers [1], [2] that match the numbered source list.
 */
object AnswerFormatter {

    enum class Kind { HEADING, PARAGRAPH, BULLET, NUMBERED }

    /** One run of text; [bold] runs are emphasised, [ref] runs are citation numbers like "[1]". */
    data class Run(val text: String, val bold: Boolean = false, val ref: Boolean = false)

    data class Block(val kind: Kind, val runs: List<Run>, val marker: String = "") {
        val plain: String get() = runs.joinToString("") { it.text }
    }

    data class Formatted(val blocks: List<Block>, val citedIndexes: List<Int>)

    private val HEADING = Regex("""^#{1,6}\s+(.*)$""")
    private val BULLET = Regex("""^[*\-•+]\s+(.*)$""")
    private val NUMBERED = Regex("""^(\d{1,2})[.)]\s+(.*)$""")
    private val CITATION = Regex("""\[([A-Za-z0-9_:\-]+(?:\s*[,;]\s*[A-Za-z0-9_:\-]+)*)]""")

    /** [sourceIds] are the evidence IDs given to the model, in the order of the source list (1-based numbers). */
    fun format(answer: String, sourceIds: List<String> = emptyList()): Formatted {
        val cited = LinkedHashSet<Int>()
        val blocks = ArrayList<Block>()
        val paragraph = StringBuilder()

        fun flushParagraph() {
            if (paragraph.isNotBlank()) blocks += Block(Kind.PARAGRAPH, inline(paragraph.toString().trim(), sourceIds, cited))
            paragraph.clear()
        }

        for (raw in answer.replace("\r", "").lines()) {
            val line = raw.trim()
            if (line.isEmpty() || line.matches(Regex("""^[-*_`]{3,}$"""))) {
                flushParagraph()
                continue
            }
            val h = HEADING.find(line)
            val b = BULLET.find(line)
            val n = NUMBERED.find(line)
            when {
                h != null -> { flushParagraph(); blocks += Block(Kind.HEADING, inline(h.groupValues[1], sourceIds, cited).map { it.copy(bold = !it.ref) }) }
                b != null -> { flushParagraph(); blocks += Block(Kind.BULLET, inline(b.groupValues[1], sourceIds, cited), "•") }
                n != null -> { flushParagraph(); blocks += Block(Kind.NUMBERED, inline(n.groupValues[2], sourceIds, cited), "${n.groupValues[1]}.") }
                else -> { if (paragraph.isNotEmpty()) paragraph.append(' '); paragraph.append(line) }
            }
        }
        flushParagraph()
        return Formatted(blocks.filter { it.plain.isNotBlank() }, cited.toList())
    }

    /** The answer as plain sentences for read-aloud: no citation numbers, bullets or Markdown. */
    fun spokenText(answer: String): String =
        format(answer).blocks.joinToString("\n") { b ->
            val text = b.runs.filterNot { it.ref }.joinToString("") { it.text }.trim()
            val withStep = if (b.kind == Kind.NUMBERED) "${b.marker} $text" else text
            if (withStep.isNotEmpty() && withStep.last() in ".?!:।") withStep else "$withStep."
        }

    /** Inline pass: citations -> [n], **bold** / __bold__ -> bold runs, leftover markup removed. */
    private fun inline(text: String, sourceIds: List<String>, cited: MutableSet<Int>): List<Run> {
        // 1. Citations. Known IDs become numbers; unknown ones (hallucinated) are dropped.
        val withRefs = CITATION.replace(text) { m ->
            val nums = m.groupValues[1].split(',', ';').map { it.trim() }
                .mapNotNull { id -> sourceIds.indexOf(id).takeIf { it >= 0 }?.plus(1) }
                .distinct().sorted()
            if (nums.isEmpty()) "" else { cited += nums; "\u0000" + nums.joinToString(",") + "\u0000" }
        }
        // 2. Bold markers, then strip stray Markdown.
        val runs = ArrayList<Run>()
        val parts = withRefs.split(Regex("""\*\*|__"""))
        parts.forEachIndexed { i, part ->
            // Odd segments sit between ** markers; an unclosed final ** does not make the rest bold.
            val bold = i % 2 == 1 && i < parts.size - 1
            val pieces = part.split('\u0000')
            pieces.forEachIndexed { k, piece ->
                if (k % 2 == 1) {
                    // "record [1,2]": exactly one space between the text and its reference number.
                    if (runs.isNotEmpty() && !runs.last().ref) runs[runs.size - 1] = runs.last().let { it.copy(text = it.text.trimEnd()) }
                    runs += Run(" [$piece]", ref = true)
                } else {
                    val clean = piece.replace("`", "").replace(Regex("""(?<![\w*])\*(?!\s)([^*]+?)\*(?!\w)"""), "$1").replace("*", "")
                    if (clean.isNotEmpty()) runs += Run(clean, bold = bold)
                }
            }
        }
        // Tidy spacing: no space before punctuation, single spaces.
        return runs.map { r ->
            if (r.ref) r else r.copy(text = r.text.replace(Regex("""\s+"""), " ").replace(Regex("""\s+([.,;:!?])"""), "$1"))
        }
            .mergeAdjacent()
            .trimEdges()
    }

    private fun List<Run>.mergeAdjacent(): List<Run> {
        val out = ArrayList<Run>()
        for (r in this) {
            val last = out.lastOrNull()
            if (last != null && last.bold == r.bold && last.ref == r.ref) out[out.size - 1] = last.copy(text = last.text + r.text) else out += r
        }
        return out
    }

    private fun List<Run>.trimEdges(): List<Run> {
        if (isEmpty()) return this
        val out = toMutableList()
        out[0] = out[0].copy(text = out[0].text.trimStart())
        out[out.size - 1] = out.last().copy(text = out.last().text.trimEnd())
        return out.filter { it.text.isNotEmpty() }
    }
}
