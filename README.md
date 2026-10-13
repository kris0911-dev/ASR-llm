# ASR-llm: Korean speech-to-text demo

Tests **BuzzASR/korean** (Whisper-large-v3 fine-tuned for Korean only) on your own voice or wav files.

## Setup (once)
Open PowerShell in this folder:
```
C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\pip install -r requirements.txt
```
First run downloads the model (~3 GB) into the Hugging Face cache.

## Run
- Double-click `run.bat`
- **Start recording** → speak Korean → **Stop**. The wav is saved in `recordings\` and the text appears with the time taken.
- **Transcribe file...** → pick a wav to test existing audio.
- Tick **compare with openai/whisper-small** to see a baseline (downloads ~1 GB extra the first time).

Without the window: `run.bat --file test.wav` (add `--baseline` to compare).

## Notes
- Uses the GPU (CUDA) if available, otherwise CPU (much slower).
- One clip is up to 30 seconds; longer audio is cut.
- Korean only.
