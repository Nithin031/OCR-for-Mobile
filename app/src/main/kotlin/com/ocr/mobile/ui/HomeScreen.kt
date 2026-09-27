package com.ocr.mobile.ui

import android.app.Activity
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.TextButton
import androidx.compose.ui.text.font.FontFamily
import com.ocr.mobile.GOLDEN_RUNNING
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.runtime.collectAsState
import com.ocr.core.EngineRegistry
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import com.google.mlkit.vision.documentscanner.GmsDocumentScanning
import com.google.mlkit.vision.documentscanner.GmsDocumentScannerOptions
import com.google.mlkit.vision.documentscanner.GmsDocumentScanningResult
import com.ocr.mobile.MainViewModel

@Composable
fun HomeScreen(viewModel: MainViewModel) {
    val context = LocalContext.current
    val activity = context as Activity

    val snackbarHostState = remember { SnackbarHostState() }
    var pendingError by remember { mutableStateOf<String?>(null) }

    // Show errors emitted by the ViewModel (image load failures, asset issues).
    LaunchedEffect(Unit) {
        viewModel.errorMessage.collect { msg ->
            snackbarHostState.showSnackbar(msg)
        }
    }

    // Show scanner-launch errors surfaced locally.
    LaunchedEffect(pendingError) {
        pendingError?.let {
            snackbarHostState.showSnackbar(it)
            pendingError = null
        }
    }

    // System photo picker (no permission required on API 26+).
    val photoPickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.PickVisualMedia()
    ) { uri ->
        uri?.let { viewModel.loadImageFromUri(context, it) }
    }

    // ML Kit Document Scanner launched via IntentSender.
    val scannerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.StartIntentSenderForResult()
    ) { result ->
        if (result.resultCode == Activity.RESULT_OK) {
            val scanResult = GmsDocumentScanningResult.fromActivityResultIntent(result.data)
            val uri = scanResult?.pages?.firstOrNull()?.imageUri
            if (uri != null) {
                viewModel.loadImageFromUri(context, uri)
            } else {
                pendingError = "No page returned from scanner"
            }
        }
    }

    fun launchScanner() {
        val options = GmsDocumentScannerOptions.Builder()
            .setGalleryImportAllowed(true)
            .setPageLimit(1)
            .setResultFormats(GmsDocumentScannerOptions.RESULT_FORMAT_JPEG)
            .setScannerMode(GmsDocumentScannerOptions.SCANNER_MODE_FULL)
            .build()

        GmsDocumentScanning.getClient(options)
            .getStartScanIntent(activity)
            .addOnSuccessListener { intentSender ->
                scannerLauncher.launch(IntentSenderRequest.Builder(intentSender).build())
            }
            .addOnFailureListener { e ->
                Log.w("HomeScreen", "Document scanner unavailable", e)
                pendingError = "Document scanner not available on this device — use \"Pick image from gallery\" instead"
            }
    }

    GoldenDialog(viewModel)
    Scaffold(snackbarHost = { SnackbarHost(snackbarHostState) }) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 32.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Text(
                text = "OCR for Mobile",
                style = MaterialTheme.typography.headlineMedium,
            )
            Spacer(Modifier.height(24.dp))
            EngineSelector(viewModel)
            Spacer(Modifier.height(24.dp))
            Button(
                onClick = { launchScanner() },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Scan document")
            }
            Spacer(Modifier.height(16.dp))
            OutlinedButton(
                onClick = {
                    photoPickerLauncher.launch(
                        PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)
                    )
                },
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text("Pick image from gallery")
            }
            if (viewModel.goldenAvailable) {
                Spacer(Modifier.height(16.dp))
                TextButton(onClick = { viewModel.runGoldenCheck() }) {
                    Text("Run golden check (debug, selected engine)")
                }
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
private fun EngineSelector(viewModel: MainViewModel) {
    val selected by viewModel.engineId.collectAsState()
    Text("OCR engine", style = MaterialTheme.typography.labelLarge)
    Spacer(Modifier.height(8.dp))
    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        EngineRegistry.entries.forEach { entry ->
            FilterChip(
                selected = entry.id == selected,
                onClick = {
                    viewModel.selectEngine(entry.id)
                    viewModel.warmUp(entry.id)
                },
                label = { Text(entry.displayName) },
            )
        }
    }
}

@Composable
fun GoldenDialog(viewModel: MainViewModel) {
    val text by viewModel.golden.collectAsState()
    val report = text ?: return
    AlertDialog(
        onDismissRequest = { viewModel.dismissGolden() },
        confirmButton = {
            if (report != GOLDEN_RUNNING) TextButton(onClick = { viewModel.dismissGolden() }) { Text("Close") }
        },
        title = { Text("Golden check") },
        text = {
            SelectionContainer {
                Text(
                    report,
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                    modifier = Modifier.verticalScroll(rememberScrollState()),
                )
            }
        },
    )
}
