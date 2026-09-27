package com.ocr.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FieldExtractorTest {

    private fun extract(vararg rows: List<String>) = FieldExtractor.extract(rows.toList())

    @Test
    fun labelAndValueInOneLine() {
        val f = extract(listOf("Name: Ramesh Kumar")).single()
        assertEquals(FieldKind.TEXT, f.kind)
        assertEquals("Name", f.label)
        assertEquals("Ramesh Kumar", f.value)
    }

    @Test
    fun labelLineFollowedByValueInSameRow() {
        val f = extract(listOf("Father's Name:", "Suresh Kumar")).single()
        assertEquals("Father's Name", f.label)
        assertEquals("Suresh Kumar", f.value)
    }

    @Test
    fun labelledValueTakesFormatKind() {
        val f = extract(listOf("Date of Birth: 05/11/1990")).single()
        assertEquals(FieldKind.DATE, f.kind)
        assertEquals("Date of Birth", f.label)
        assertEquals("05/11/1990", f.value)
    }

    @Test
    fun unlabelledFormatsAreFound() {
        val fields = extract(
            listOf("PAN ABCDE1234F"),
            listOf("Mob 9876543210 email ram.k@example.in"),
            listOf("Aadhaar 2345 6789 0123"),
            listOf("IFSC SBIN0001234"),
            listOf("Fee Rs. 1,500.00"),
        )
        val byKind = fields.associate { it.kind to it.value }
        assertEquals("ABCDE1234F", byKind[FieldKind.PAN])
        assertEquals("9876543210", byKind[FieldKind.PHONE])
        assertEquals("ram.k@example.in", byKind[FieldKind.EMAIL])
        assertEquals("2345 6789 0123", byKind[FieldKind.AADHAAR])
        assertEquals("SBIN0001234", byKind[FieldKind.IFSC])
        assertEquals("Rs. 1,500.00", byKind[FieldKind.AMOUNT])
    }

    @Test
    fun aadhaarIsNotAlsoReportedAsPhone() {
        val kinds = extract(listOf("2345 6789 0123")).map { it.kind }
        assertEquals(listOf(FieldKind.AADHAAR), kinds)
    }

    @Test
    fun pinCodeOnlyNearPinKeyword() {
        assertTrue(extract(listOf("Order 560001 shipped")).none { it.kind == FieldKind.PINCODE })
        assertEquals("560001", extract(listOf("PIN 560001")).single { it.kind == FieldKind.PINCODE }.value)
    }

    @Test
    fun implausibleDatesAndTimesAreIgnored() {
        assertTrue(extract(listOf("ref 45/78/2020")).none { it.kind == FieldKind.DATE })
        assertTrue(extract(listOf("10:30")).isEmpty())
        assertTrue(extract(listOf("https://example.gov.in")).none { it.kind == FieldKind.TEXT })
    }

    @Test
    fun monthNameDate() {
        assertEquals("12 Jan 2024", extract(listOf("Issued on 12 Jan 2024")).single { it.kind == FieldKind.DATE }.value)
    }

    @Test
    fun valuesAreNotCorrected() {
        // OCR errors (O for 0) are kept as read.
        assertEquals("ABCDE12O4F", extract(listOf("PAN No: ABCDE12O4F")).single().value)
    }
}
