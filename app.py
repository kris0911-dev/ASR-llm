"""Minimal Korean ASR demo using BuzzASR/korean (Whisper-large-v3 fine-tune).

GUI:      python app.py
Headless: python app.py --file some.wav [--baseline]
"""
import argparse, os, sys, threading, time, datetime
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

SR = 16000  # model expects 16 kHz mono
MAIN_MODEL = "BuzzASR/korean"
BASELINE_MODEL = "openai/whisper-small"  # optional, downloaded only when used (~1 GB)
HERE = os.path.dirname(os.path.abspath(__file__))
REC_DIR = os.path.join(HERE, "recordings")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
_cache = {}


def load(name):
    if name not in _cache:
        t = time.time()
        proc = WhisperProcessor.from_pretrained(name)
        model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=DTYPE).to(DEVICE).eval()
        _cache[name] = (model, proc)
        print(f"[load] {name} on {DEVICE} in {time.time()-t:.1f}s")
    return _cache[name]


def read_audio(path):
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # to mono
    if sr != SR:  # simple linear resample
        n = int(len(wav) * SR / sr)
        wav = np.interp(np.linspace(0, len(wav), n, endpoint=False), np.arange(len(wav)), wav).astype("float32")
    return wav


def transcribe(wav, name=MAIN_MODEL):
    model, proc = load(name)
    t = time.time()
    feats = proc(wav, sampling_rate=SR, return_tensors="pt").input_features.to(DEVICE, DTYPE)
    kw = dict(num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2)
    if name != MAIN_MODEL:  # BuzzASR has Korean baked in; baseline needs it set
        kw.update(language="ko", task="transcribe")
    with torch.inference_mode():
        ids = model.generate(feats, **kw)
    text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
    return text, time.time() - t
# Note: Whisper handles up to 30 s per call; longer audio is cut at 30 s.


def run_gui():
    import tkinter as tk
    from tkinter import filedialog
    import sounddevice as sd

    FONT = ("Malgun Gothic", 14)
    root = tk.Tk(); root.title("Korean ASR demo (BuzzASR)"); root.geometry("900x650")
    state = {"rec": False, "chunks": [], "stream": None, "busy": False}
    use_base = tk.BooleanVar(value=False)
    status = tk.StringVar(value="Loading model...")

    def set_status(s):
        status.set(s); print("[status]", s)

    def show_result(header, lines):
        # latest result box (replaced each time)
        latest.config(state="normal"); latest.delete("1.0", "end")
        latest.insert("end", header + "\n\n" + "\n\n".join(lines)); latest.config(state="disabled")
        # history: newest on top
        hist.config(state="normal")
        hist.insert("1.0", header + "\n" + "\n".join(lines) + "\n" + "-" * 60 + "\n")
        hist.config(state="disabled")

    def process(path):
        if state["busy"]:
            set_status("Busy - wait for the current transcription."); return
        state["busy"] = True
        root.after(0, set_status, "Transcribing...")
        def work():
            try:
                wav = read_audio(path)
                header = f"{datetime.datetime.now():%H:%M:%S}  {os.path.basename(path)}  ({len(wav)/SR:.1f}s audio)"
                lines = []
                names = [MAIN_MODEL] + ([BASELINE_MODEL] if use_base.get() else [])
                for n in names:
                    label = "BuzzASR/korean" if n == MAIN_MODEL else f"Baseline {n}"
                    text, dt = transcribe(wav, n)
                    print(f"[{label}] {dt:.2f}s: {text}")
                    lines.append(f"[{label}]  ({dt:.2f}s)\n{text}")
                root.after(0, show_result, header, lines)
                root.after(0, set_status, "Done")
            except Exception as e:
                root.after(0, set_status, f"ERROR: {e}")
            finally:
                state["busy"] = False
        threading.Thread(target=work, daemon=True).start()

    def toggle():
        if not state["rec"]:
            state["chunks"] = []
            state["stream"] = sd.InputStream(samplerate=SR, channels=1, dtype="float32",
                                             callback=lambda d, f, t, s: state["chunks"].append(d.copy()))
            state["stream"].start(); state["rec"] = True
            btn.config(text="Stop"); set_status("Recording... (speak Korean, then press Stop)")
        else:
            state["stream"].stop(); state["stream"].close(); state["rec"] = False
            btn.config(text="Start recording")
            if not state["chunks"]:
                set_status("No audio captured."); return
            os.makedirs(REC_DIR, exist_ok=True)
            path = os.path.join(REC_DIR, datetime.datetime.now().strftime("rec_%Y%m%d_%H%M%S.wav"))
            sf.write(path, np.concatenate(state["chunks"])[:, 0], SR)
            print("[saved]", path); process(path)

    def open_file():
        p = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.flac *.ogg"), ("All", "*.*")])
        if p: process(p)

    bar = tk.Frame(root); bar.pack(fill="x", padx=6, pady=6)
    btn = tk.Button(bar, text="Start recording", width=16, font=("Segoe UI", 11), command=toggle); btn.pack(side="left")
    tk.Button(bar, text="Transcribe file...", font=("Segoe UI", 11), command=open_file).pack(side="left", padx=6)
    tk.Checkbutton(bar, text=f"Also compare with {BASELINE_MODEL}", variable=use_base).pack(side="left")
    tk.Label(root, textvariable=status, anchor="w", font=("Segoe UI", 11, "bold"), fg="#1f4e79").pack(fill="x", padx=6)
    tk.Label(root, text="Latest result", anchor="w").pack(fill="x", padx=6)
    latest = tk.Text(root, height=8, wrap="word", font=FONT, bg="#f4f7fb", state="disabled")
    latest.pack(fill="x", padx=6)
    tk.Label(root, text="History (newest on top)", anchor="w").pack(fill="x", padx=6, pady=(8, 0))
    hf = tk.Frame(root); hf.pack(fill="both", expand=True, padx=6, pady=(0, 6))
    hist = tk.Text(hf, wrap="word", font=("Malgun Gothic", 12), state="disabled")
    sb = tk.Scrollbar(hf, command=hist.yview); hist.config(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y"); hist.pack(side="left", fill="both", expand=True)

    dev = DEVICE + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "")
    set_status(f"Loading model on {dev} (first run downloads ~3 GB)...")
    def preload():
        try:
            load(MAIN_MODEL); root.after(0, set_status, f"Ready on {dev}. Press Start recording.")
        except Exception as e:
            root.after(0, set_status, f"ERROR loading model: {e}")
    threading.Thread(target=preload, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", help="transcribe this audio file without GUI")
    ap.add_argument("--baseline", action="store_true", help=f"also run {BASELINE_MODEL}")
    a = ap.parse_args()
    if a.file:
        sys.stdout.reconfigure(encoding="utf-8")
        wav = read_audio(a.file)
        print(f"audio: {len(wav)/SR:.1f}s, device: {DEVICE}")
        for n in [MAIN_MODEL] + ([BASELINE_MODEL] if a.baseline else []):
            text, dt = transcribe(wav, n)
            print(f"[{n}] {dt:.2f}s: {text}")
    else:
        run_gui()
