package com.ocr.mobile

import com.ocr.mobile.AnswerFormatter.Kind
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class AnswerFormatterTest {

    @Test
    fun bulletsWithBoldLabelsAreClean() {
        val f = AnswerFormatter.format("*   **Mobile Number:** Some information\n* **PAN:** Withholding tax")
        assertEquals(listOf(Kind.BULLET, Kind.BULLET), f.blocks.map { it.kind })
        val first = f.blocks[0]
        assertEquals("Mobile Number: Some information", first.plain)
        assertTrue(first.runs.first().bold)
        assertFalse(first.plain.contains("*"))
    }

    @Test
    fun citationsBecomeNumbersAndUnknownOnesAreDropped() {
        val ids = listOf("fact_pmfby", "sf-abc:3")
        val f = AnswerFormatter.format("You need a land record [fact_pmfby, sf-abc:3]. Claim in 72 hours [fact_made_up].", ids)
        assertEquals("You need a land record [1,2]. Claim in 72 hours.", f.blocks.single().plain)
        assertEquals(listOf(1, 2), f.citedIndexes)
    }

    @Test
    fun numberedStepsHeadingsAndParagraphs() {
        val f = AnswerFormatter.format("## How to apply\n1. Visit the portal\n2) Upload documents\n\nThat is all.\nThanks.")
        assertEquals(listOf(Kind.HEADING, Kind.NUMBERED, Kind.NUMBERED, Kind.PARAGRAPH), f.blocks.map { it.kind })
        assertEquals("2.", f.blocks[2].marker)
        assertEquals("That is all. Thanks.", f.blocks[3].plain)
    }

    @Test
    fun unbalancedMarkupIsStripped() {
        val f = AnswerFormatter.format("This is **important and `code` here")
        assertEquals("This is important and code here", f.blocks.single().plain)
        assertFalse(f.blocks.single().runs.any { it.bold })
    }
}
