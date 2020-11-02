# TensorFlow TTS

TensorFlowTTS provides real-time state-of-the-art speech synthesis architectures such as Tacotron-2, Melgan, Multiband-Melgan, FastSpeech, FastSpeech2 based-on TensorFlow 2. 

It supports English, Korean, and Chinese.

- Feature_generator models
    - Tacotron2
    - FastSpeech2
    - FastSpeech

- Vocoder models
    - MB-MelGAN
    - MelGAN-STFT
    - MelGAN

|         | Tacotron2 | FastSpeech2 | FastSpeech | MB-MelGAN | MelGAN-STFT | MelGAN |
|---------|:---------:|:-----------:|:----------:|:---------:|:-----------:|:------:|
| English |     O     |      O      |      O     |     O     |      O      |    O   |
| Korean  |     O     |      O      |      X     |     O     |      X      |    X   |
| Chinese |     O     |      O      |      X     |     O     |      X      |    X   |

## Curl Example
- language: english, korean, chinese

- feature_generator: TACOTRON2, FASTSPEECH2, FASTSPEECH

- vocoder: MB-MELGAN, MELGAN-STFT, MELGAN

```bash
curl -o save_name.wav -X POST "https://server-tensor-flow-tts-psi1104.endpoint.ainize.ai/predict" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "language=english" -F "feature_generator=TACOTRON2" -F "vocoder=MB-MELGAN" -F "input_text=This is test sentence."
```
