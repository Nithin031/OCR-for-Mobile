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
import com.ocr.mobile.AnswerFormatter
import com.ocr.mobile.ReadAloud
import com.ocr.mobile.R
import androidx.compose.ui.res.stringResource
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
    val (tts, ttsState) = rememberReadAloud()
    val ocr by viewModel.ocrState.collectAsState()
    val ai by viewModel.aiState.collectAsState()

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.result)) },
                navigationIcon = {
                    IconButton(onClick = { viewModel.goHome() }) {
                        Icon(imageVector = Icons.AutoMirrored.Filled.ArrowBack, contentDescription = stringResource(R.string.back))
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
                text = stringResource(R.string.image_info, bitmap.width, bitmap.height),
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier
                    .align(Alignment.CenterHorizontally)
                    .padding(vertical = 4.dp),
            )
            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
            OcrSection(ocr, onRunMlKit = { viewModel.rerun(EngineRegistry.FALLBACK_ID) }) { text ->
                ReadAloudButton(text, key = "ocr", tts = tts, state = ttsState)
            }
            if (ocr.fields.isNotEmpty()) {
                HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
                FieldsSection(ocr.fields)
            }
            HorizontalDivider(modifier = Modifier.padding(vertical = 8.dp))
            AssistantSection(ocr, ai, viewModel, tts, ttsState)
            Spacer(Modifier.height(24.dp))
        }
    }
}
@Composable
private fun FieldsSection(fields: List<DocField>) {
    Text(stringResource(R.string.detected_fields), style = MaterialTheme.typography.titleMedium)
    Text(
        stringResource(R.string.fields_note),
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
                    Text(stringResource(R.string.more_fields, fields.size - MAX_FIELDS_SHOWN), style = MaterialTheme.typography.bodySmall)
                }
            }
        }
    }
}

private const val MAX_FIELDS_SHOWN = 40

@Composable
private fun AssistantSection(ocr: OcrUi, ai: AiUi, viewModel: MainViewModel, tts: ReadAloud, ttsState: ReadAloudState) {
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) viewModel.importModel(uri)
    }
    var question by remember { mutableStateOf("") }
    val (voice, voiceState) = rememberVoiceQuestion()

    val modelName = ai.activeModel?.let { GemmaEngine.friendlyName(it) } ?: "Gemma"
    Text(stringResource(R.string.ask_ai_title, modelName), style = MaterialTheme.typography.titleMedium)
    val kagCounts = ai.kagCounts
    if (ai.kagStatus.isNotEmpty() || kagCounts != null) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(
                checked = ai.useKnowledge && ai.kagReady,
                enabled = ai.kagReady && !ai.busy,
                onCheckedChange = { viewModel.setUseKnowledge(it) },
            )
            Column {
                Text(stringResource(R.string.use_kag), style = MaterialTheme.typography.bodyMedium)
                Text(
                    kagCounts?.let { (docs, passages, schemes) -> stringResource(R.string.kag_ready, docs, passages, schemes) }
                        ?: ai.kagStatus,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary,
                )
            }
        }
    }
    Spacer(Modifier.height(8.dp))

    val progress = ai.importProgress
    when {
        progress != null -> {
            Text(stringResource(R.string.copying_model, (progress * 100).toInt()))
            Spacer(Modifier.height(4.dp))
            LinearProgressIndicator(progress = { progress }, modifier = Modifier.fillMaxWidth())
        }
        !ai.modelPresent -> {
            Text(
                stringResource(R.string.no_model),
                style = MaterialTheme.typography.bodyMedium,
            )
            Spacer(Modifier.height(8.dp))
            Button(onClick = { picker.launch(arrayOf("*/*")) }) { Text(stringResource(R.string.import_model)) }
        }
        else -> {
            val canAsk = !ai.busy && ocr.lines.isNotEmpty()
            ModelPicker(ai, onSelect = { viewModel.selectModel(it) }, onImport = { picker.launch(arrayOf("*/*")) })
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(enabled = canAsk, onClick = { viewModel.ask(DocPrompt.SUMMARIZE) }) { Text(stringResource(R.string.summarize)) }
                OutlinedButton(enabled = canAsk, onClick = { viewModel.ask(DocPrompt.ASKS_FOR) }) { Text(stringResource(R.string.asks_for)) }
            }
            Spacer(Modifier.height(8.dp))
            OutlinedTextField(
                value = question,
                onValueChange = { question = it },
                label = { Text(stringResource(R.string.question_hint)) },
                modifier = Modifier.fillMaxWidth(),
                trailingIcon = {
                    VoiceQuestionButton(voice, voiceState, enabled = !ai.busy, onText = { question = it })
                },
            )
            VoiceQuestionStatus(voice, voiceState)
            Spacer(Modifier.height(8.dp))
            Button(enabled = canAsk && question.isNotBlank(), onClick = { viewModel.ask(question) }) { Text(stringResource(R.string.ask)) }
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
                Text(stringResource(R.string.question_label, ai.question.orEmpty()), style = MaterialTheme.typography.labelLarge)
                Spacer(Modifier.height(6.dp))
                FormattedAnswer(ai.answer, ai.sources)
                ReadAloudButton(AnswerFormatter.spokenText(ai.answer), key = "answer", tts = tts, state = ttsState)
                Spacer(Modifier.height(8.dp))
                Text(
                    stringResource(R.string.ai_footer, ai.linesUsed, ai.linesTotal, "%.1f".format(ai.answerSeconds ?: 0.0)),
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary,
                )
                if (ai.questionLanguage != "en") {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        stringResource(R.string.answer_in_english_note),
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.secondary,
                    )
                }
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
        TextButton(enabled = !ai.busy, onClick = onImport) { Text(stringResource(R.string.import_another_model)) }
    }
    Spacer(Modifier.height(4.dp))
}
