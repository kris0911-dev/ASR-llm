# ASR-llm for Android (BuzzASR Korean, offline, on the phone)

This app records your voice, then turns Korean speech into text **on the phone**, with no internet.
It uses **BuzzASR/korean** (Whisper large-v3 fine-tuned for Korean), shrunk ("quantized") so it fits on a phone,
and runs it with **whisper.cpp**.

Screen: status line, **Start/Stop recording**, **Transcribe WAV file...**, **Pick model file...**,
the latest result in big text with its timing, and a history list.

## What you need
- A **64-bit** Android phone (almost every phone since 2015), **Android 5.0 or newer**.
- Free RAM: about **1.5 GB** for the q5_0 model, about **2.2 GB** for q8_0. Phones with 4 GB+ RAM are best.
- Free storage: about 1.1 GB (q5_0) or 1.7 GB (q8_0).

## Model files (in `..\android-model\`)
| File | Size | Notes |
|---|---|---|
| `ggml-buzz-ko-q5_0.bin` | 1.08 GB | smaller, recommended for most phones |
| `ggml-buzz-ko-q8_0.bin` | 1.66 GB | not included (saves disk). Make it with: `tools\whisper-quantize.exe <f16 file> ggml-buzz-ko-q8_0.bin q8_0` |

The model is **not** inside the app (it is too big). You copy it to the phone once (step 3).

## 1. Open in Android Studio
1. Start Android Studio -> **Open** -> choose this `android` folder.
2. Wait for "Gradle sync" to finish. If it asks to install missing parts (Android SDK 35, **NDK 27.2.12479018**,
   **CMake 3.22.1**), click the install links. (Or: *Tools -> SDK Manager -> SDK Tools*, tick *NDK (Side by side)* and *CMake*.)

## 2. Build and install
- Plug in the phone with USB debugging on (Settings -> About phone -> tap *Build number* 7 times -> Developer options -> *USB debugging*).
- Press the green **Run** button. Or build only: *Build -> Build App Bundle(s)/APK(s) -> Build APK(s)*.
  The APK appears in `app\build\outputs\apk\debug\app-debug.apk`.
- A ready-made APK is also in `prebuilt\app-debug.apk`.

## 3. Copy the model to the phone (pick one way)
**A. With a USB cable and adb (fastest)** — open the app once first, then in a terminal:
```
adb push ..\android-model\ggml-buzz-ko-q5_0.bin /sdcard/Android/data/com.asrllm/files/
```
(adb is in `%LOCALAPPDATA%\Android\Sdk\platform-tools\`.)

**B. Without a computer at hand** — copy the `.bin` file to the phone's Downloads (USB, cloud, etc.),
then in the app tap **Pick model file (.bin)...** and choose it. The app copies it into its own folder.

Only keep **one** `.bin` in the app folder (the app uses the first one, in name order).

## 4. Use it
- Tap **Start recording**, speak Korean, tap **Stop recording**. The first time it loads the model (can take 10-60 s).
- The text appears in big letters, with the audio length and how long transcription took.
- **Transcribe WAV file...** works on `.wav` files (any sample rate, mono or stereo).
- Recording stops by itself after 120 s.

## Benchmark (how fast is it on my phone?)
1. First record something or transcribe a WAV (the benchmark re-uses that last audio).
2. Tap **Benchmark (last audio, 2/4/6/8 threads)**. It transcribes the same audio once per thread count
   (so it takes 3-4x as long as one transcription).
3. The table shows, per thread count: total seconds, **RTF** (real-time factor = processing time / audio length;
   below 1.0 means faster than real time), and the stages: encode, decode, batch-decode, sampling, prompt.
4. The top lines show phone model, cores, CPU flags (asimddp = dotprod, fphp = fp16, i8mm),
   and which ggml CPU build was picked.
5. **Copy benchmark report** copies it; it is also saved as `benchmark-<time>.txt` next to the model
   (`/sdcard/Android/data/com.asrllm/files/`, `adb pull` it).

Normal transcriptions use the "default threads" shown there (the phone's fast cores).

## Good to know
- Speed depends on the phone. Whisper large-v3 is a big model: expect several seconds to tens of seconds
  for a short sentence on mid-range phones. It uses up to 4 CPU cores, no GPU.
- The app is built for 64-bit phones only (arm64-v8a, plus x86_64 for the emulator). 32-bit phones cannot load a 1 GB+ model.
- The app contains several builds of the math library (ARMv8.0 baseline, ARMv8.2 dotprod/fp16, ARMv8.6 i8mm, ARMv9 SVE/SME). At start it picks the fastest one your CPU supports, so old phones still work.
- `whisper.cpp\` contains the needed part of whisper.cpp (MIT licence), see `whisper.cpp\VERSION.txt`.

## How the model files were made (for reference)
The tools are in `..\android-model\tools\` (Windows) and `tools\` here.
1. Used `BuzzASR/korean` (`model.safetensors`, F16) from the Hugging Face cache.
2. Converted to GGML F16 with `python tools\conv.py out-f16.bin <model folder> tools\mel_filters.npz` (equivalent to whisper.cpp `models/convert-h5-to-ggml.py`
   ; mel filters `tools/mel_filters.npz` from openai/whisper).
3. Quantized with whisper.cpp: `whisper-quantize ggml-f16.bin ggml-buzz-ko-q5_0.bin q5_0` (and `q8_0`).
