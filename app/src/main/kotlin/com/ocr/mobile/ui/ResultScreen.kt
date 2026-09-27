package com.ocr.mobile.ui

import android.graphics.Bitmap
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import com.ocr.core.DocField
import com.ocr.core.FieldKind
import com.ocr.mobile.AiUi
import com.ocr.mobile.DocPrompt
import com.ocr.mobile.MainViewModel
import com.ocr.mobile.OcrUi

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ResultScreen(bitmap: Bitmap, viewModel: MainViewModel) {
    val ocr by viewModel.ocrState.collectAsState()
    val ai by viewModel.aiState.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Result") },
                navigationIcon = {
                    IconButton(onClick = { viewModel.goHome() }) {
                        Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = "Back")
                    }
                },
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
        ) {
            Image(
                bitmap = bitmap.asImageBitmap(),
                contentDescription = "Captured document",
                modifier = Modifier
                    .fillMaxWidth()
                    .height(280.dp),
                contentScale = ContentScale.Fit,
            )
            Text(
                text = "${bitmap.width} × ${bitmap.height} px",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier
                    .align(Alignment.CenterHorizontally)
                    .padding(vertical = 4.dp),
            )
            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
            OcrSection(ocr)
            if (ocr.fields.isNotEmpty()) {
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                FieldsSection(ocr.fields)
            }
            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
            AssistantSection(ocr, ai, viewModel)
            Spacer(Modifier.height(24.dp))
        }
    }
}

@Composable
private fun OcrSection(ocr: OcrUi) {
    Text("Recognized text", style = MaterialTheme.typography.titleMedium)
    Spacer(Modifier.height(8.dp))
    when {
        ocr.running -> Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
            Spacer(Modifier.width(8.dp))
            Text("Reading text on the phone…")
        }
        ocr.error != null -> Text(ocr.error, color = MaterialTheme.colorScheme.error)
        ocr.lines.isEmpty() -> Text("No text found in this image.")
        else -> {
            Text(
                "${ocr.lines.size} lines · ${ocr.elapsedMs} ms (ML Kit, offline)",
                style = MaterialTheme.typography.bodySmall,
            )
            if (ocr.pipelineNote.isNotEmpty()) {
                Text(ocr.pipelineNote, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
            }
            if (ocr.lowConfidence.isNotEmpty()) {
                Text(
                    "${ocr.lowConfidence.size} unclear lines are underlined in red: check them on the paper.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
            Spacer(Modifier.height(4.dp))
            val errorColor = MaterialTheme.colorScheme.error
            val text = remember(ocr.lines, ocr.lowConfidence, errorColor) {
                buildAnnotatedString {
                    ocr.lines.forEachIndexed { i, line ->
                        if (i > 0) append('\n')
                        if (i in ocr.lowConfidence) {
                            withStyle(SpanStyle(color = errorColor, textDecoration = TextDecoration.Underline)) { append(line) }
                        } else {
                            append(line)
                        }
                    }
                }
            }
            Card(modifier = Modifier.fillMaxWidth()) {
                SelectionContainer {
                    Text(
                        text = text,
                        style = MaterialTheme.typography.bodyMedium,
                        modifier = Modifier.padding(12.dp),
                    )
                }
            }
        }
    }
}

@Composable
private fun FieldsSection(fields: List<DocField>) {
    Text("Detected fields", style = MaterialTheme.typography.titleMedium)
    Text(
        "Copied exactly as read — nothing is corrected.",
        style = MaterialTheme.typography.bodySmall,
        color = MaterialTheme.colorScheme.secondary,
    )
    Spacer(Modifier.height(4.dp))
    Card(modifier = Modifier.fillMaxWidth()) {
        SelectionContainer {
            Column(modifier = Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                fields.take(MAX_FIELDS_SHOWN).forEach { f ->
                    Column {
                        val caption = if (f.kind == FieldKind.TEXT || f.label == f.kind.label) f.label else "${f.label} · ${f.kind.label}"
                        Text(caption, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.primary)
                        Text(f.value, style = MaterialTheme.typography.bodyMedium)
                    }
                }
                if (fields.size > MAX_FIELDS_SHOWN) {
                    Text("+${fields.size - MAX_FIELDS_SHOWN} more", style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

private const val MAX_FIELDS_SHOWN = 40

@Composable
private fun AssistantSection(ocr: OcrUi, ai: AiUi, viewModel: MainViewModel) {
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) viewModel.importModel(uri)
    }
    var question by remember { mutableStateOf("") }

    Text("Ask the on-device AI (Gemma 3 1B)", style = MaterialTheme.typography.titleMedium)
    Spacer(Modifier.height(8.dp))

    val progress = ai.importProgress
    when {
        progress != null -> {
            Text("Copying the model into the app… ${(progress * 100).toInt()}%")
            Spacer(Modifier.height(4.dp))
            LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
        }
        !ai.modelPresent -> {
            Text(
                "No AI model yet. Copy gemma3-1b-it-int4.task to this phone, then import it once. " +
                    "After that everything runs offline.",
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.height(8.dp))
            Button(onClick = { picker.launch(arrayOf("*/*")) }) { Text("Import model file (.task)") }
        }
        else -> {
            val canAsk = !ai.busy && ocr.lines.isNotEmpty()
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(enabled = canAsk, onClick = { viewModel.ask(DocPrompt.SUMMARIZE) }) { Text("Summarize") }
                OutlinedButton(enabled = canAsk, onClick = { viewModel.ask(DocPrompt.ASKS_FOR) }) { Text("What does it ask for?") }
            }
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = question,
                onValueChange = { question = it },
                label = { Text("Your question about this document") },
                modifier = Modifier.fillMaxWidth(),
            )
            Spacer(Modifier.height(8.dp))
            Button(enabled = canAsk && question.isNotBlank(), onClick = { viewModel.ask(question) }) { Text("Ask") }
        }
    }

    if (ai.busy) {
        Spacer(Modifier.height(12.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
            Spacer(Modifier.width(8.dp))
            Text(ai.status)
        }
    }
    if (ai.error != null) {
        Spacer(Modifier.height(8.dp))
        Text(ai.error, color = MaterialTheme.colorScheme.error)
    }
    if (ai.answer != null) {
        Spacer(Modifier.height(12.dp))
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(12.dp)) {
                Text("Q: ${ai.question}", style = MaterialTheme.typography.labelLarge)
                Spacer(Modifier.height(6.dp))
                SelectionContainer { Text(ai.answer, style = MaterialTheme.typography.bodyMedium) }
                Spacer(Modifier.height(8.dp))
                Text(
                    "AI-generated — check against the document. " +
                        "Used ${ai.linesUsed} of ${ai.linesTotal} lines · ${"%.1f".format(ai.answerSeconds ?: 0.0)} s on this phone",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary,
                )
            }
        }
    }
}
