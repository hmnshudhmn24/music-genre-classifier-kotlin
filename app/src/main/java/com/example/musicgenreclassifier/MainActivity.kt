
package com.example.musicgenreclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : ComponentActivity() {
    private lateinit var interpreter: Interpreter

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        interpreter = Interpreter(
            assets.openFd("music_genre.tflite").createInputStream().readBytes()
        )

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.RECORD_AUDIO),
                1000
            )
        }

        setContent {
            GenreClassifierApp(interpreter)
        }
    }
}

@Composable
fun GenreClassifierApp(interpreter: Interpreter) {
    var genre by remember { mutableStateOf("Awaiting audio...") }
    var confidence by remember { mutableStateOf(0.0f) }

    LaunchedEffect(Unit) {
        val audioData = captureAudio()
        val mfccFeatures = extractMFCC(audioData)
        val prediction = classifyGenre(interpreter, mfccFeatures)
        genre = prediction.first
        confidence = prediction.second
    }

    Surface(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .padding(20.dp)
                .fillMaxSize(),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text("Predicted Genre: $genre", style = MaterialTheme.typography.headlineMedium)
            Text("Confidence: ${(confidence * 100).toInt()}%", style = MaterialTheme.typography.bodyLarge)
        }
    }
}

suspend fun captureAudio(): ShortArray {
    val sampleRate = 16000
    val bufferSize = AudioRecord.getMinBufferSize(
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT
    )
    val audioRecord = AudioRecord(
        MediaRecorder.AudioSource.MIC,
        sampleRate,
        AudioFormat.CHANNEL_IN_MONO,
        AudioFormat.ENCODING_PCM_16BIT,
        bufferSize
    )
    val audioBuffer = ShortArray(bufferSize)
    audioRecord.startRecording()
    audioRecord.read(audioBuffer, 0, bufferSize)
    audioRecord.stop()
    audioRecord.release()
    return audioBuffer
}

fun extractMFCC(audio: ShortArray): Array<FloatArray> {
    val mfcc = Array(40) { FloatArray(40) }
    for (i in mfcc.indices) {
        for (j in mfcc[i].indices) {
            mfcc[i][j] = audio.getOrNull(i * j % audio.size)?.toFloat() ?: 0f
        }
    }
    return mfcc
}

fun classifyGenre(interpreter: Interpreter, mfcc: Array<FloatArray>): Pair<String, Float> {
    val inputBuffer = ByteBuffer.allocateDirect(40 * 40 * 4).order(ByteOrder.nativeOrder())
    for (row in mfcc) {
        for (value in row) {
            inputBuffer.putFloat(value)
        }
    }
    val output = Array(1) { FloatArray(10) }
    interpreter.run(inputBuffer, output)
    val genres = listOf("rock", "pop", "jazz", "classical", "hiphop", "metal", "blues", "country", "edm", "reggae")
    val maxIndex = output[0].indices.maxByOrNull { output[0][it] } ?: 0
    return genres[maxIndex] to output[0][maxIndex]
}
