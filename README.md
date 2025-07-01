
# 🎵 Music Genre Classifier Kotlin App

A cutting-edge Android app built with Kotlin and Jetpack Compose to classify music genres from short audio samples. It uses a pre-trained TensorFlow Lite model and a simple feature-extraction pipeline directly on-device, giving you lightning-fast offline predictions with no server dependency. Sleek UI, instant feedback, and a robust modern Kotlin architecture.

## Project Highlights

- Real-time music genre prediction
- TFLite embedded inference
- Kotlin-based feature extractor
- Compose-based interface
- Lightweight and blazing fast

## How to Run

1. Export your TFLite model in Python and place it in `assets/music_genre.tflite`.
2. Clone the repo:
   ```
   git clone https://github.com/yourusername/music-genre-classifier-kotlin.git
   ```
3. Open in Android Studio and click **Run**.

## Customization

- Replace the model anytime
- Add top-3 predictions
- Expand MFCC visualization

## Next Features

- Microphone stream classification
- Batch classification
- Recommendations
