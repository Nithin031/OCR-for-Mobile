package com.ocr.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class GemmaModelNameTest {

    @Test
    fun friendlyNames() {
        assertEquals("Gemma 3 4B", GemmaEngine.friendlyName("gemma3-4b-it-int4.task"))
        assertEquals("Gemma 3 1B", GemmaEngine.friendlyName("gemma3-1b-it-int4.task"))
        assertEquals("my-model", GemmaEngine.friendlyName("my-model.task"))
    }

    @Test
    fun importKeepsModelFileNamesAndRejectsOthers() {
        assertEquals("gemma3-1b-it-int4.task", GemmaEngine.safeFileName("gemma3-1b-it-int4.task"))
        assertThrows(IllegalArgumentException::class.java) { GemmaEngine.safeFileName("gemma-4-E2B-it.litertlm") }
        assertEquals("gemma_3_copy.task", GemmaEngine.safeFileName("gemma 3 copy.task"))
        assertThrows(IllegalArgumentException::class.java) { GemmaEngine.safeFileName("photo.jpg") }
        assertThrows(IllegalArgumentException::class.java) { GemmaEngine.safeFileName("") }
    }
}
