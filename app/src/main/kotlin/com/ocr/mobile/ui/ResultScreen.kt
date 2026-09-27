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
import androidx.compose.material3.Checkbox
import androidx.compose.material3.FilterChip
import androidx.compose.material3.TextButton
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import com.ocr.mobile.GemmaEngine
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
import com.ocr.core.EngineRegistry
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
            ZoomableDocument(bitmap, ocr)
            Text(
                text = "${bitmap.width} × ${bitmap.height} px · pinch to zoom, boxes show detected lines",
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier
                    .align(Alignment.CenterHorizontally)
                    .padding(vertical = 4.dp),
            )
            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
            OcrSection(ocr, onRunMlKit = { viewModel.rerun(EngineRegistry.DEFAULT_ID) })
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

    val modelName = ai.activeModel?.let { GemmaEngine.friendlyName(it) } ?: "Gemma"
    Text("Ask the on-device AI ($modelName)", style = MaterialTheme.typography.titleMedium)
    if (ai.kagStatus.isNotEmpty()) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(
                checked = ai.useKnowledge && ai.kagReady,
                enabled = ai.kagReady && !ai.busy,
                onCheckedChange = { viewModel.setUseKnowledge(it) },
            )
            Column {
                Text("Use scheme knowledge base (offline KAG)", style = MaterialTheme.typography.bodyMedium)
                Text(ai.kagStatus, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
            }
        }
    }
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
            ModelPicker(ai, onSelect = { viewModel.selectModel(it) }, onImport = { picker.launch(arrayOf("*/*")) })
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
                FormattedAnswer(ai.answer, ai.sources)
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

/** Imported models as chips (tap to switch) plus a way to import another .task file. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun ModelPicker(ai: AiUi, onSelect: (String) -> Unit, onImport: () -> Unit) {
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.Center) {
        if (ai.models.size > 1) {
            ai.models.forEach { name ->
                FilterChip(
                    selected = name == ai.activeModel,
                    enabled = !ai.busy,
                    onClick = { onSelect(name) },
                    label = { Text(GemmaEngine.friendlyName(name)) },
                )
            }
        }
        TextButton(enabled = !ai.busy, onClick = onImport) { Text("Import another model") }
    }
    Spacer(Modifier.height(4.dp))
}
