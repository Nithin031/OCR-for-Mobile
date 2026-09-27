package com.ocr.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SpeechTest {

    @Test
    fun speechLanguageFollowsAppLanguage() {
        assertEquals("en-IN", SpeechLanguages.tagFor("en"))
        assertEquals("hi-IN", SpeechLanguages.tagFor("hi"))
        assertEquals("kn-IN", SpeechLanguages.tagFor("kn"))
        assertEquals("en-IN", SpeechLanguages.tagFor("fr"))
    }

    @Test
    fun readAloudVoiceFollowsTheTextsScript() {
        assertEquals("hi-IN", SpeechLanguages.tagForText("फसल बीमा के लिए दस्तावेज़"))
        assertEquals("kn-IN", SpeechLanguages.tagForText("ಬೆಳೆ ವಿಮೆಗೆ ಯಾವ ದಾಖಲೆಗಳು ಬೇಕು"))
        assertEquals("en-IN", SpeechLanguages.tagForText("You need a land record."))
        assertEquals("en-IN", SpeechLanguages.tagForText("PMFBY: land record (RTC) and bank passbook"))
    }

    @Test
    fun spokenAnswerHasNoCitationsOrMarkup() {
        val spoken = AnswerFormatter.spokenText("**Documents:**\n* Land record [fact_pmfby_docs]\n1. Visit the portal")
        assertFalse(spoken.contains("["))
        assertFalse(spoken.contains("*"))
        assertEquals("Documents:\nLand record.\n1. Visit the portal.", spoken)
    }

    @Test
    fun longTextIsSplitAtSentenceEnds() {
        val text = "First sentence. Second one here! पहला वाक्य। Last?"
        val parts = ReadAloud.chunks(text, 20)
        assertTrue(parts.all { it.length <= 20 })
        assertEquals(text.replace(Regex("""\s+"""), " "), parts.joinToString(" "))
    }
}
