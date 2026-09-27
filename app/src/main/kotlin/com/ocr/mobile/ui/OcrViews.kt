package com.ocr.mobile.ui

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import com.ocr.core.EngineRegistry
import com.ocr.core.OcrResult
import com.ocr.core.UNKNOWN_CONFIDENCE
import com.ocr.mobile.OcrUi
import kotlin.math.min

private val BoxOk = Color(0xFF1E88E5)
private val BoxLow = Color(0xFFE53935)

/** Document image with the detected line boxes drawn on top; pinch to zoom, drag to pan. */
@Composable
fun ZoomableDocument(bitmap: Bitmap, ocr: OcrUi) {
    var scale by remember(bitmap) { mutableFloatStateOf(1f) }
    var offset by remember(bitmap) { mutableStateOf(Offset.Zero) }
    val image = remember(bitmap) { bitmap.asImageBitmap() }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(340.dp)
            .clipToBounds()
            .pointerInput(bitmap) {
                detectTransformGestures { _, pan, zoom, _ ->
                    scale = (scale * zoom).coerceIn(1f, MAX_ZOOM)
                    offset = if (scale == 1f) Offset.Zero else offset + pan
                }
            },
    ) {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .graphicsLayer(scaleX = scale, scaleY = scale, translationX = offset.x, translationY = offset.y),
        ) {
            Image(bitmap = image, contentDescription = "Captured document", modifier = Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
            Canvas(modifier = Modifier.fillMaxSize()) {
                // Same mapping as ContentScale.Fit, centred.
                val s = min(size.width / bitmap.width, size.height / bitmap.height)
                val dx = (size.width - bitmap.width * s) / 2f
                val dy = (size.height - bitmap.height * s) / 2f
                val stroke = Stroke(width = 1.5.dp.toPx() / scale)
                ocr.ocrLines.forEachIndexed { i, line ->
                    val c = line.corners ?: with(line.box) {
                        floatArrayOf(
                            left.toFloat(), top.toFloat(), right.toFloat(), top.toFloat(),
                            right.toFloat(), bottom.toFloat(), left.toFloat(), bottom.toFloat(),
                        )
                    }
                    val path = Path().apply {
                        moveTo(dx + c[0] * s, dy + c[1] * s)
                        for (k in 1 until 4) lineTo(dx + c[2 * k] * s, dy + c[2 * k + 1] * s)
                        close()
                    }
                    drawPath(path, if (i in ocr.lowConfidence) BoxLow else BoxOk, style = stroke)
                }
            }
        }
    }
}

@Composable
fun OcrSection(ocr: OcrUi, onRunMlKit: () -> Unit) {
    Text("Recognized text", style = MaterialTheme.typography.titleMedium)
    if (ocr.engineName.isNotEmpty()) {
        Text("Engine: ${ocr.engineName} · offline", style = MaterialTheme.typography.bodySmall)
    }
    Spacer(Modifier.height(8.dp))
    when {
        ocr.running -> Row(verticalAlignment = Alignment.CenterVertically) {
            CircularProgressIndicator(modifier = Modifier.size(20.dp), strokeWidth = 2.dp)
            Spacer(Modifier.width(8.dp))
            Text(ocr.stage.ifEmpty { "Reading text on the phone…" })
        }
        ocr.error != null -> {
            Text(ocr.error, color = MaterialTheme.colorScheme.error)
            if (ocr.engineId != EngineRegistry.DEFAULT_ID) {
                Spacer(Modifier.height(8.dp))
                Button(onClick = onRunMlKit) { Text("Run with ML Kit") }
            }
        }
        ocr.ocrLines.isEmpty() -> Text("No text found in this image.")
        else -> {
            Text("${ocr.ocrLines.size} lines · ${ocr.elapsedMs} ms total", style = MaterialTheme.typography.bodySmall)
            if (ocr.pipelineNote.isNotEmpty()) {
                Text(ocr.pipelineNote, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.secondary)
            }
            TimingsTable(ocr.timings)
            if (ocr.lowConfidence.isNotEmpty()) {
                Text(
                    "${ocr.lowConfidence.size} lines below ${"%.2f".format(OcrResult.LOW_CONFIDENCE)} confidence are " +
                        "highlighted: check them on the paper.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
            Spacer(Modifier.height(4.dp))
            Card(modifier = Modifier.fillMaxWidth()) {
                SelectionContainer {
                    Column(modifier = Modifier.padding(8.dp)) {
                        val lowBg = MaterialTheme.colorScheme.errorContainer
                        ocr.ocrLines.forEachIndexed { i, line ->
                            val low = i in ocr.lowConfidence
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .then(if (low) Modifier.background(lowBg) else Modifier)
                                    .padding(horizontal = 4.dp, vertical = 2.dp),
                            ) {
                                Text(
                                    if (line.confidence == UNKNOWN_CONFIDENCE) " n/a" else "%.2f".format(line.confidence),
                                    style = MaterialTheme.typography.bodySmall,
                                    fontFamily = FontFamily.Monospace,
                                    color = if (low) MaterialTheme.colorScheme.error else MaterialTheme.colorScheme.secondary,
                                    modifier = Modifier.width(44.dp),
                                )
                                Text(line.text, style = MaterialTheme.typography.bodyMedium)
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun TimingsTable(timings: List<Pair<String, Long>>) {
    if (timings.isEmpty()) return
    Column(modifier = Modifier.padding(vertical = 4.dp)) {
        timings.forEach { (stage, ms) ->
            Row {
                Text(stage, style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                Text("$ms ms", style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
            }
        }
    }
}

private const val MAX_ZOOM = 8f
