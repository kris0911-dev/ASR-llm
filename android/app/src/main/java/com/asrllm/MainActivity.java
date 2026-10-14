package com.asrllm;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.content.ActivityNotFoundException;
import android.os.Environment;
import android.os.StatFs;
import android.view.View;
import android.view.WindowInsets;
import android.widget.Button;
import android.widget.Toast;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Minimal functional demo: record -> BuzzASR (Whisper large-v3 Korean, quantized GGML) -> text. */
public class MainActivity extends Activity {
    private static final int SR = 16000;              // Whisper needs 16 kHz mono
    private static final int MAX_SECONDS = 120;
    private static final int REQ_MIC = 1, REQ_WAV = 2, REQ_MODEL = 3;
    private static final String LANG = "ko";

    private final ExecutorService worker = Executors.newSingleThreadExecutor(); // whisper: one thread at a time
    private final Handler ui = new Handler(Looper.getMainLooper());

    private TextView status, result, timing, history;
    private Button recBtn, wavBtn, modelBtn;
    private long ctx = 0;
    private String loadedModel = null;
    private volatile boolean recording = false;
    private volatile boolean busy = false;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        int pad = dp(12);
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setPadding(pad, pad, pad, pad);

        TextView title = new TextView(this); title.setText("ASR-llm (BuzzASR Korean)"); title.setTextSize(20);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        status = new TextView(this); status.setTextSize(14); status.setTextColor(0xFF1565C0);
        status.setPadding(0, pad / 2, 0, pad / 2);
        recBtn = new Button(this); recBtn.setText("Start recording");
        wavBtn = new Button(this); wavBtn.setText("Transcribe WAV file...");
        modelBtn = new Button(this); modelBtn.setText("Pick model file (.bin)...");
        result = new TextView(this); result.setTextSize(24); result.setTypeface(Typeface.DEFAULT_BOLD);
        result.setPadding(0, pad, 0, 0); result.setTextIsSelectable(true);
        timing = new TextView(this); timing.setTextSize(13);
        TextView hTitle = new TextView(this); hTitle.setText("\nHistory"); hTitle.setTypeface(Typeface.DEFAULT_BOLD);
        history = new TextView(this); history.setTextSize(15); history.setTextIsSelectable(true);

        col.addView(title); col.addView(status); col.addView(recBtn); col.addView(wavBtn); col.addView(modelBtn);
        col.addView(result); col.addView(timing); col.addView(hTitle); col.addView(history);
        ScrollView sv = new ScrollView(this); sv.addView(col);
        sv.setFillViewport(true);
        // Android 15+ draws apps edge-to-edge: pad content so the status/navigation bars don't cover it.
        if (Build.VERSION.SDK_INT >= 20) {
            sv.setOnApplyWindowInsetsListener((v, in) -> {
                int top, bottom, left, right;
                if (Build.VERSION.SDK_INT >= 30) {
                    android.graphics.Insets i = in.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                    top = i.top; bottom = i.bottom; left = i.left; right = i.right;
                } else { top = in.getSystemWindowInsetTop(); bottom = in.getSystemWindowInsetBottom(); left = in.getSystemWindowInsetLeft(); right = in.getSystemWindowInsetRight(); }
                v.setPadding(left, top, right, bottom);
                return in;
            });
        }
        setContentView(sv);

        recBtn.setOnClickListener(v -> { if (recording) recording = false; else startRecording(); });
        wavBtn.setOnClickListener(v -> pick("audio/*", REQ_WAV));
        modelBtn.setOnClickListener(v -> pick("*/*", REQ_MODEL));

        if (Build.VERSION.SDK_INT >= 23 && checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 0);
        File m = findModel();
        if (m == null) setStatus("No model found.\nTap 'Pick model file' and choose ggml-buzz-ko-q5_0.bin,\nor adb push it to:\n" + modelDir());
        else setStatus("Model: " + m.getAbsolutePath() + " (" + (m.length() >> 20) + " MB, loads on first use)");
    }

    // ---------- model ----------
    private File modelDir() { File d = getExternalFilesDir(null); return d != null ? d : getFilesDir(); }

    private File findModel() {
        File f = firstBin(modelDir(), false);
        if (f != null) return f;
        try { // readable only on old Android or with broad storage access; harmless otherwise
            File dl = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS);
            f = firstBin(dl, true);
            if (f != null && f.canRead()) return f;
        } catch (Throwable ignored) {}
        return null;
    }

    private static File firstBin(File dir, boolean ggmlOnly) {
        if (dir == null) return null;
        File[] fs = dir.listFiles((d, n) -> n.toLowerCase().endsWith(".bin") && (!ggmlOnly || n.toLowerCase().contains("ggml")));
        if (fs == null || fs.length == 0) return null;
        Arrays.sort(fs);
        return fs[0];
    }

    /** Runs on worker thread. */
    private boolean ensureModel() {
        File m = findModel();
        if (m == null) { setStatus("No model file in " + modelDir()); return false; }
        if (ctx != 0 && m.getAbsolutePath().equals(loadedModel)) return true;
        if (ctx != 0) { Whisper.free(ctx); ctx = 0; }
        setStatus("Loading " + m.getName() + " (" + (m.length() >> 20) + " MB)...");
        long t0 = System.currentTimeMillis();
        ctx = Whisper.init(m.getAbsolutePath());
        if (ctx == 0) { toastStatus("Failed to load model " + m.getName() + " (not enough RAM or bad file)."); return false; }
        loadedModel = m.getAbsolutePath();
        setStatus("Model loaded in " + (System.currentTimeMillis() - t0) / 1000.0 + " s: " + m.getName());
        return true;
    }

    // ---------- recording ----------
    private void startRecording() {
        if (busy) return;
        if (Build.VERSION.SDK_INT >= 23 && checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQ_MIC);
            return;
        }
        recording = true;
        recBtn.setText("Stop recording");
        setStatus("Recording... (max " + MAX_SECONDS + " s)");
        new Thread(() -> {
            int minBuf = AudioRecord.getMinBufferSize(SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
            AudioRecord rec = new AudioRecord(MediaRecorder.AudioSource.MIC, SR,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, Math.max(minBuf, SR * 2));
            if (rec.getState() != AudioRecord.STATE_INITIALIZED) {
                recording = false; setStatus("Microphone not available."); ui.post(() -> recBtn.setText("Start recording")); return;
            }
            short[] buf = new short[SR / 10];
            ByteArrayOutputStream pcm = new ByteArrayOutputStream();
            rec.startRecording();
            while (recording && pcm.size() < SR * 2 * MAX_SECONDS) {
                int n = rec.read(buf, 0, buf.length);
                for (int i = 0; i < n; i++) { pcm.write(buf[i] & 0xff); pcm.write((buf[i] >> 8) & 0xff); }
            }
            rec.stop(); rec.release();
            recording = false;
            ui.post(() -> recBtn.setText("Start recording"));
            byte[] raw = pcm.toByteArray();
            float[] f = new float[raw.length / 2];
            ByteBuffer bb = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
            for (int i = 0; i < f.length; i++) f[i] = bb.getShort() / 32768f;
            transcribe(f, "mic");
        }).start();
    }

    @Override public void onRequestPermissionsResult(int code, String[] p, int[] r) {
        if (code == REQ_MIC && r.length > 0 && r[0] == PackageManager.PERMISSION_GRANTED) startRecording();
        else if (code == REQ_MIC) toastStatus("Microphone permission denied. Allow it in Settings > Apps.");
    }

    // ---------- transcription ----------
    private void transcribe(float[] pcm, String label) {
        busy = true;
        worker.execute(() -> {
            try {
                if (!ensureModel()) return;
                double sec = pcm.length / (double) SR;
                setStatus("Transcribing " + String.format("%.1f", sec) + " s of audio...");
                int threads = Math.max(1, Math.min(4, Runtime.getRuntime().availableProcessors()));
                long t0 = System.currentTimeMillis();
                String text = Whisper.transcribe(ctx, pcm, threads, LANG).trim();
                double took = (System.currentTimeMillis() - t0) / 1000.0;
                String tm = String.format("%s | audio %.1f s | transcribe %.2f s | %d threads", label, sec, took, threads);
                ui.post(() -> {
                    CharSequence old = result.getText();
                    if (old.length() > 0) history.setText(old + "\n  (" + timing.getText() + ")\n\n" + history.getText());
                    result.setText(text.isEmpty() ? "(no speech recognized)" : text);
                    timing.setText(tm);
                });
                setStatus("Done.");
            } catch (Throwable t) {
                toastStatus("Error: " + t);
            } finally { busy = false; }
        });
    }

    // ---------- files ----------
    private void pick(String mime, int req) {
        if (busy || recording) { toastStatus("Busy - wait until the current task finishes."); return; }
        String[] actions = Build.VERSION.SDK_INT >= 19
                ? new String[]{Intent.ACTION_OPEN_DOCUMENT, Intent.ACTION_GET_CONTENT}
                : new String[]{Intent.ACTION_GET_CONTENT};
        for (String a : actions) {
            Intent i = new Intent(a);
            i.setType(mime);
            i.addCategory(Intent.CATEGORY_OPENABLE);
            try {
                startActivityForResult(i, req);
                setStatus("Choose a file...");
                return;
            } catch (ActivityNotFoundException | SecurityException e) {
                // try next action
            }
        }
        toastStatus("No file picker app found on this phone. Use adb push instead (see README).");
    }

    private void toastStatus(String s) {
        setStatus(s);
        ui.post(() -> Toast.makeText(this, s, Toast.LENGTH_LONG).show());
    }

    @Override protected void onActivityResult(int req, int res, Intent data) {
        if (res != RESULT_OK || data == null || data.getData() == null) { setStatus("No file chosen."); return; }
        Uri uri = data.getData();
        if (req == REQ_WAV) {
            busy = true;
            worker.execute(() -> {
                try (InputStream in = getContentResolver().openInputStream(uri)) {
                    float[] f = Wav.read16kMono(in);
                    busy = false;
                    transcribe(f, "file");
                } catch (Throwable t) { busy = false; toastStatus("Cannot read WAV: " + t); }
            });
        } else if (req == REQ_MODEL) {
            busy = true;
            worker.execute(() -> {
                File tmp = new File(modelDir(), "incoming.tmp");
                long size = -1;
                try (android.content.res.AssetFileDescriptor afd = getContentResolver().openAssetFileDescriptor(uri, "r")) {
                    if (afd != null) size = afd.getLength();
                } catch (Throwable ignored) {}
                try {
                    long free = new StatFs(modelDir().getAbsolutePath()).getAvailableBytes();
                    if (size > 0 && size + (50L << 20) > free) {
                        toastStatus("Not enough storage: need " + (size >> 20) + " MB, free " + (free >> 20) + " MB.");
                        busy = false; return;
                    }
                } catch (Throwable ignored) {}
                final long total0 = size;
                try (InputStream in = getContentResolver().openInputStream(uri); OutputStream out = new FileOutputStream(tmp)) {
                    if (in == null) throw new java.io.IOException("cannot open the chosen file");
                    byte[] buf = new byte[1 << 20]; long total = 0, lastMb = -1; int n;
                    while ((n = in.read(buf)) > 0) {
                        out.write(buf, 0, n); total += n;
                        long mb = total >> 20;
                        if (mb / 20 != lastMb) { lastMb = mb / 20; setStatus("Copying model... " + mb + (total0 > 0 ? " / " + (total0 >> 20) : "") + " MB"); }
                    }
                    out.close();
                    if (total < (10L << 20)) throw new java.io.IOException("file too small (" + total + " bytes) - not a model");
                    File[] old = modelDir().listFiles((d, nm) -> nm.toLowerCase().endsWith(".bin"));
                    if (old != null) for (File o : old) o.delete();
                    if (ctx != 0) { Whisper.free(ctx); ctx = 0; loadedModel = null; }
                    File dst = new File(modelDir(), "model.bin");
                    if (!tmp.renameTo(dst) && !dst.exists()) throw new java.io.IOException("rename failed");
                    setStatus("Model copied (" + (dst.length() >> 20) + " MB). Loading...");
                    ensureModel();
                } catch (Throwable t) { tmp.delete(); toastStatus("Model copy failed: " + t); }
                finally { busy = false; }
            });
        }
    }

    private void setStatus(String s) { ui.post(() -> status.setText(s)); }
    private int dp(int v) { return (int) (v * getResources().getDisplayMetrics().density); }

    @Override protected void onDestroy() {
        recording = false;
        worker.execute(() -> { if (ctx != 0) { Whisper.free(ctx); ctx = 0; } });
        worker.shutdown();
        super.onDestroy();
    }
}
