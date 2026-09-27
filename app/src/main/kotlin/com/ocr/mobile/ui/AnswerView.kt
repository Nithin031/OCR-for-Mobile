package com.ocr.mobile.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.ocr.mobile.AnswerFormatter
import com.ocr.mobile.KagSource

/** The model answer as clean text: headings, bullets, numbered steps, bold, and [n] source numbers. */
@Composable
fun FormattedAnswer(answer: String, sources: List<KagSource>) {
    val formatted = remember(answer, sources) { AnswerFormatter.format(answer, sources.map { it.id }) }
    val refColor = MaterialTheme.colorScheme.primary

    SelectionContainer {
        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
            formatted.blocks.forEach { block ->
                val text = remember(block, refColor) { block.toAnnotated(refColor) }
                when (block.kind) {
                    AnswerFormatter.Kind.HEADING ->
                        Text(text, style = MaterialTheme.typography.titleSmall)
                    AnswerFormatter.Kind.PARAGRAPH ->
                        Text(text, style = MaterialTheme.typography.bodyMedium)
                    AnswerFormatter.Kind.BULLET, AnswerFormatter.Kind.NUMBERED -> Row {
                        Text(block.marker, style = MaterialTheme.typography.bodyMedium, modifier = Modifier.width(22.dp))
                        Text(text, style = MaterialTheme.typography.bodyMedium)
                    }
                }
            }
        }
    }

    if (sources.isNotEmpty()) {
        // Show the sources the answer cites; if it cites none, the evidence it was given.
        val shown = formatted.citedIndexes.ifEmpty { sources.indices.map { it + 1 }.take(MAX_UNCITED_SOURCES) }
        Spacer(Modifier.height(10.dp))
        Text(
            if (formatted.citedIndexes.isNotEmpty()) "Sources" else "Evidence given to the AI",
            style = MaterialTheme.typography.labelMedium,
        )
        shown.forEach { n ->
            val s = sources[n - 1]
            Text(
                "[$n] ${s.title}${s.url?.let { " — $it" }.orEmpty()}",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.secondary,
            )
        }
    }
}

private fun AnswerFormatter.Block.toAnnotated(refColor: Color): AnnotatedString = buildAnnotatedString {
    runs.forEach { r ->
        when {
            r.ref -> withStyle(SpanStyle(color = refColor, fontSize = 12.sp, fontWeight = FontWeight.Medium)) { append(r.text) }
            r.bold -> withStyle(SpanStyle(fontWeight = FontWeight.SemiBold)) { append(r.text) }
            else -> append(r.text)
        }
    }
}

private const val MAX_UNCITED_SOURCES = 6
