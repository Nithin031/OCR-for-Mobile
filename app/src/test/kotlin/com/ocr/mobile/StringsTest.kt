package com.ocr.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/** Guards the multilingual UI: English text is unchanged, and Hindi/Kannada match it key by key. */
class StringsTest {

    private fun load(dir: String): Map<String, String> {
        val file = listOf(File("src/main/res/$dir/strings.xml"), File("app/src/main/res/$dir/strings.xml")).first { it.isFile }
        return Regex("""<string name="([^"]+)">(.*?)</string>""", RegexOption.DOT_MATCHES_ALL)
            .findAll(file.readText())
            .associate { it.groupValues[1] to it.groupValues[2].replace("\\\"", "\"").replace("%%", "%") }
    }

    private fun placeholders(s: String) = Regex("""%\d+\$[ds]""").findAll(s).map { it.value }.toSortedSet()

    @Test
    fun englishLabelsAreExactlyTheOriginalOnes() {
        val en = load("values")
        // The literals that were in the code before the multilingual change, with sample arguments.
        val expected = mapOf(
            "scan_document" to "Scan document",
            "pick_image" to "Pick image from gallery",
            "ocr_engine" to "OCR engine",
            "scanner_no_page" to "No page returned from scanner",
            "scanner_unavailable" to "Document scanner not available on this device — use \"Pick image from gallery\" instead",
            "result" to "Result",
            "recognized_text" to "Recognized text",
            "run_with_mlkit" to "Run with ML Kit",
            "no_text_found" to "No text found in this image.",
            "detected_fields" to "Detected fields",
            "fields_note" to "Copied exactly as read — nothing is corrected.",
            "use_kag" to "Use scheme knowledge base (offline KAG)",
            "no_model" to "No AI model yet. Copy gemma3-1b-it-int4.task to this phone, then import it once. After that everything runs offline.",
            "import_model" to "Import model file (.task)",
            "summarize" to "Summarize",
            "asks_for" to "What does it ask for?",
            "question_hint" to "Your question about this document",
            "ask" to "Ask",
            "status_thinking" to "Thinking… (on this phone, can take up to a minute)",
        )
        expected.forEach { (key, text) -> assertEquals(key, text, en[key]) }
        assertEquals("78 lines · 1411 ms total", String.format(en.getValue("lines_total"), 78, 1411L))
        assertEquals("Ask the on-device AI (Gemma 3 1B)", String.format(en.getValue("ask_ai_title"), "Gemma 3 1B"))
        assertEquals(
            "AI-generated — check against the document. Used 46 of 78 lines · 14.5 s on this phone",
            String.format(en.getValue("ai_footer"), 46, 78, "14.5"),
        )
        assertEquals(
            "12 lines below 0.80 confidence are highlighted: check them on the paper.",
            String.format(en.getValue("low_confidence_note"), 12, "0.80"),
        )
        assertEquals("100 documents · 2241 passages · 4 schemes in graph", String.format(en.getValue("kag_ready"), 100, 2241, 4))
    }

    @Test
    fun hindiAndKannadaHaveEveryKeyWithTheSamePlaceholders() {
        val en = load("values")
        for (dir in listOf("values-hi", "values-kn")) {
            val other = load(dir)
            assertEquals("$dir keys", en.keys, other.keys)
            en.forEach { (key, text) ->
                assertEquals("$dir/$key placeholders", placeholders(text), placeholders(other.getValue(key)))
                assertTrue("$dir/$key is empty", other.getValue(key).isNotBlank())
            }
        }
    }
}
