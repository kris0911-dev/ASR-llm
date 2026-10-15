package com.asrllm;

import android.Manifest;
import android.animation.ObjectAnimator;
import android.animation.PropertyValuesHolder;
import android.animation.ValueAnimator;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.Color;
import android.graphics.Typeface;
import android.graphics.drawable.Drawable;
import android.graphics.drawable.GradientDrawable;
import android.graphics.drawable.RippleDrawable;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.Looper;
import android.os.StatFs;
import android.view.Gravity;
import android.view.View;
import android.view.Window;
import android.view.WindowInsets;
import android.widget.FrameLayout;
import android.widget.HorizontalScrollView;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.Switch;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/** Record -> BuzzASR (Whisper large-v3 Korean, quantized GGML via whisper.cpp) -> text. Optional live mode. */
public class MainActivity extends Activity {
    // ---- audio / model ----
    private static final int SR = 16000;              // Whisper needs 16 kHz mono
    private static final int MAX_SECONDS = 600;       // hard cap 10 min
    private static final int LEVEL_WIN = 480;         // 30 ms per waveform bar
    private static final int REQ_MIC = 1, REQ_WAV = 2, REQ_MODEL = 3;
    private static final String LANG = "ko";

    // ---- palette: soft mid-tone slate blue + muted coral ----
    static final int C_BG = 0xFFE6EBF0, C_CARD = 0xFFF6F8FA, C_PRIMARY = 0xFF56789A, C_PRIMARY_SOFT = 0xFFD5E0EA,
            C_REC = 0xFFD2685F, C_TEXT = 0xFF2F3B47, C_SUB = 0xFF6A7885, C_LINE = 0xFFCBD5DE,
            C_OK = 0xFF5E9A78, C_WARN = 0xFFC9963F;

    private final ExecutorService worker = Executors.newSingleThreadExecutor(); // whisper: one call at a time
    private final Handler ui = new Handler(Looper.getMainLooper());
    private final int defaultThreads = CpuInfo.defaultThreads();

    private long ctx = 0;
    private String loadedModel = null;
    private double lastLoadSec = 0;
    private volatile double lastEnsureLoadMs = -1;     // -1: model was already loaded in the last ensureModel()
    private volatile double stopPressedAt = 0;
    private final java.util.List<Latency> liveLats = java.util.Collections.synchronizedList(new java.util.ArrayList<Latency>());
    private TextView latencyOut;
    private String lastLatencyReport = "";
    private float[] lastAudio = null;
    private volatile boolean recording = false;
    private volatile boolean busy = false;

    // ---- views ----
    private TextView status, modelChip, timerText, liveInfo, resultText, resultLabel, emuNote, benchOut, detailsArrowText;
    private LinearLayout statsRow, historyList, detailsBody;
    private FrameLayout recBtn;
    private ImageView recIcon;
    private View pulse;
    private ValueAnimator pulseAnim;
    private WaveformView wave;
    private Switch liveSwitch;
    private long recStartMs;
    private final Runnable timerTick = new Runnable() {
        @Override public void run() {
            if (!recording) return;
            long s = (System.currentTimeMillis() - recStartMs) / 1000;
            timerText.setText(String.format(Locale.ROOT, "%02d:%02d", s / 60, s % 60));
            ui.postDelayed(this, 250);
        }
    };

    // ---- recording buffer (shared with live mode) ----
    private final Object audioLock = new Object();
    private short[] rec = new short[SR * 30];
    private int recLen = 0;

    // ---- live mode state (segment queue) ----
    private final StringBuilder committed = new StringBuilder();
    private volatile int segQueued = 0, segDone = 0;   // segments sent to / finished by the worker
    private volatile int lastDoneEnd = 0;              // sample index where the last finished segment ended
    private volatile double liveInferSec = 0;

    // ---- threads setting ----
    private android.content.SharedPreferences prefs;
    private LinearLayout threadRow;
    private TextView useFastestBtn;
    private int fastestThreads = 0;

    @Override protected void onCreate(Bundle b) {
        super.onCreate(b);
        Window w = getWindow();
        if (Build.VERSION.SDK_INT >= 21) { w.setStatusBarColor(C_BG); w.setNavigationBarColor(C_BG); }
        if (Build.VERSION.SDK_INT >= 23) w.getDecorView().setSystemUiVisibility(View.SYSTEM_UI_FLAG_LIGHT_STATUS_BAR
                | (Build.VERSION.SDK_INT >= 26 ? View.SYSTEM_UI_FLAG_LIGHT_NAVIGATION_BAR : 0));
        prefs = getSharedPreferences("settings", MODE_PRIVATE);
        setContentView(buildUi());

        worker.execute(() -> {
            try {
                Whisper.loadBackends(getApplicationInfo().nativeLibraryDir);
                String info = deviceInfo();
                ui.post(() -> benchOut.setText(info));
            } catch (Throwable t) { toastStatus("Backend load error: " + t); }
        });
        if (Build.VERSION.SDK_INT >= 23 && checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, 0);
        File m = findModel();
        if (m == null) {
            setModelChip("No model", C_WARN);
            setStatus("Tap “Model” and choose ggml-buzz-ko-q4_0.bin (or q5_0/q8_0), or adb push it to " + modelDir());
        } else {
            setModelChip(m.getName() + " · " + (m.length() >> 20) + " MB · not loaded", C_SUB);
            setStatus("Ready. Tap the microphone to record.");
        }
        if (isEmulatorAbi()) emuNote.setVisibility(View.VISIBLE);
    }

    // =====================================================================================
    // UI
    // =====================================================================================
    private View buildUi() {
        int p = dp(16);
        LinearLayout col = new LinearLayout(this);
        col.setOrientation(LinearLayout.VERTICAL);
        col.setPadding(p, dp(8), p, p);

        // ---- header ----
        LinearLayout header = new LinearLayout(this);
        header.setOrientation(LinearLayout.VERTICAL);
        header.setPadding(dp(4), dp(8), dp(4), dp(12));
        TextView title = text("Korean Voice to Text", 24, C_TEXT, true);
        TextView sub = text("BuzzASR · Whisper large-v3 · on-device", 13, C_SUB, false);
        modelChip = chip("", C_PRIMARY_SOFT, C_TEXT);
        LinearLayout.LayoutParams mlp = wrap(); mlp.topMargin = dp(8);
        header.addView(title); header.addView(sub); header.addView(modelChip, mlp);
        col.addView(header);

        // ---- recorder card ----
        LinearLayout recCard = card();
        status = text("", 13, C_SUB, false);
        status.setGravity(Gravity.CENTER_HORIZONTAL);
        wave = new WaveformView(this);
        wave.setColors(C_REC, C_PRIMARY);

        FrameLayout btnWrap = new FrameLayout(this);
        int big = dp(84);
        pulse = new View(this);
        pulse.setBackground(oval(C_REC));
        pulse.setAlpha(0f);
        recBtn = new FrameLayout(this);
        recBtn.setBackground(ripple(oval(C_PRIMARY), true));
        recBtn.setElevation(dp(6));
        recIcon = new ImageView(this);
        recIcon.setImageDrawable(icon(R.drawable.ic_mic, Color.WHITE));
        recBtn.addView(recIcon, new FrameLayout.LayoutParams(dp(36), dp(36), Gravity.CENTER));
        btnWrap.addView(pulse, new FrameLayout.LayoutParams(big, big, Gravity.CENTER));
        btnWrap.addView(recBtn, new FrameLayout.LayoutParams(big, big, Gravity.CENTER));
        btnWrap.setClipChildren(false);
        LinearLayout.LayoutParams bw = new LinearLayout.LayoutParams(big + dp(40), big + dp(40));
        bw.gravity = Gravity.CENTER_HORIZONTAL; bw.topMargin = dp(4);

        timerText = text("00:00", 20, C_TEXT, true);
        timerText.setGravity(Gravity.CENTER_HORIZONTAL);
        timerText.setTypeface(Typeface.create("sans-serif-light", Typeface.NORMAL));
        liveInfo = text("", 12, C_SUB, false);
        liveInfo.setGravity(Gravity.CENTER_HORIZONTAL);
        liveInfo.setVisibility(View.GONE);

        LinearLayout liveRow = new LinearLayout(this);
        liveRow.setGravity(Gravity.CENTER_VERTICAL);
        LinearLayout liveTexts = new LinearLayout(this); liveTexts.setOrientation(LinearLayout.VERTICAL);
        liveTexts.addView(text("Live mode", 15, C_TEXT, true));
        liveTexts.addView(text("Show text while you speak (uses more CPU)", 12, C_SUB, false));
        liveSwitch = new Switch(this);
        if (Build.VERSION.SDK_INT >= 23) {
            int[][] st = {{android.R.attr.state_checked}, {}};
            liveSwitch.setThumbTintList(new ColorStateList(st, new int[]{C_PRIMARY, 0xFFF0F2F4}));
            liveSwitch.setTrackTintList(new ColorStateList(st, new int[]{0xFF9FB4C8, C_LINE}));
        }
        liveRow.addView(liveTexts, new LinearLayout.LayoutParams(0, -2, 1));
        liveRow.addView(liveSwitch);

        LinearLayout actions = new LinearLayout(this);
        actions.setGravity(Gravity.CENTER_HORIZONTAL);
        TextView wavBtn = pill("WAV file", R.drawable.ic_audio_file);
        TextView modelBtn = pill("Model", R.drawable.ic_model);
        LinearLayout.LayoutParams pl = wrap(); pl.setMargins(dp(4), 0, dp(4), 0);
        actions.addView(wavBtn, pl);
        actions.addView(modelBtn, new LinearLayout.LayoutParams(pl));

        recCard.addView(wave, new LinearLayout.LayoutParams(-1, -2));
        recCard.addView(btnWrap, bw);
        recCard.addView(timerText);
        recCard.addView(liveInfo);
        LinearLayout.LayoutParams sp = new LinearLayout.LayoutParams(-1, -2); sp.topMargin = dp(6);
        recCard.addView(status, sp);
        recCard.addView(divider());
        recCard.addView(liveRow);
        recCard.addView(divider());
        recCard.addView(text("Threads (CPU cores used)", 15, C_TEXT, true));
        threadRow = new LinearLayout(this);
        HorizontalScrollView ths = new HorizontalScrollView(this);
        ths.setHorizontalScrollBarEnabled(false);
        ths.addView(threadRow);
        LinearLayout.LayoutParams tl = new LinearLayout.LayoutParams(-1, -2); tl.topMargin = dp(6);
        recCard.addView(ths, tl);
        buildThreadRow();
        recCard.addView(divider());
        recCard.addView(actions);
        col.addView(recCard, cardLp());

        // ---- result card ----
        LinearLayout resCard = card();
        resultLabel = text("RESULT", 11, C_SUB, true);
        resultLabel.setLetterSpacing(0.1f);
        resultText = text("Your transcript will appear here.", 22, C_SUB, false);
        resultText.setLineSpacing(0, 1.3f);
        resultText.setTextIsSelectable(true);
        resultText.setPadding(0, dp(8), 0, dp(10));
        statsRow = new LinearLayout(this);
        HorizontalScrollView hs = new HorizontalScrollView(this);
        hs.setHorizontalScrollBarEnabled(false);
        hs.addView(statsRow);
        emuNote = text("Running on emulator/PC CPU (x86) — not phone speed.", 12, C_WARN, false);
        emuNote.setVisibility(View.GONE);
        resCard.addView(resultLabel); resCard.addView(resultText); resCard.addView(hs); resCard.addView(emuNote);
        col.addView(resCard, cardLp());

        // ---- latency details (collapsible) ----
        LinearLayout latCard = card();
        LinearLayout latHead = new LinearLayout(this);
        latHead.setGravity(Gravity.CENTER_VERTICAL);
        ImageView latArrow = new ImageView(this);
        latArrow.setImageDrawable(icon(R.drawable.ic_expand, C_SUB));
        latHead.addView(text("Latency details", 15, C_TEXT, true), new LinearLayout.LayoutParams(0, -2, 1));
        latHead.addView(latArrow, new LinearLayout.LayoutParams(dp(24), dp(24)));
        final LinearLayout latBody = new LinearLayout(this);
        latBody.setOrientation(LinearLayout.VERTICAL);
        latBody.setVisibility(View.GONE);
        latencyOut = text("No transcription yet.", 11, C_TEXT, false);
        latencyOut.setTypeface(Typeface.MONOSPACE);
        latencyOut.setTextIsSelectable(true);
        HorizontalScrollView ls = new HorizontalScrollView(this);
        ls.addView(latencyOut);
        TextView latCopy = pill("Copy", R.drawable.ic_copy);
        latCopy.setOnClickListener(v -> {
            android.content.ClipboardManager cm = (android.content.ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
            cm.setPrimaryClip(android.content.ClipData.newPlainText("ASR latency", latencyOut.getText()));
            Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show();
        });
        LinearLayout.LayoutParams lbl = new LinearLayout.LayoutParams(-1, -2); lbl.topMargin = dp(10);
        latBody.addView(ls, lbl);
        latBody.addView(latCopy, new LinearLayout.LayoutParams(-2, -2));
        latCard.addView(latHead); latCard.addView(latBody);
        col.addView(latCard, cardLp());
        latHead.setOnClickListener(v -> {
            boolean open = latBody.getVisibility() != View.VISIBLE;
            latBody.setVisibility(open ? View.VISIBLE : View.GONE);
            latArrow.setImageDrawable(icon(open ? R.drawable.ic_collapse : R.drawable.ic_expand, C_SUB));
        });

        // ---- history ----
        TextView hTitle = text("History", 15, C_TEXT, true);
        hTitle.setPadding(dp(4), dp(10), 0, dp(6));
        col.addView(hTitle);
        historyList = new LinearLayout(this);
        historyList.setOrientation(LinearLayout.VERTICAL);
        TextView empty = text("Nothing yet.", 13, C_SUB, false);
        empty.setPadding(dp(4), 0, 0, dp(8));
        historyList.addView(empty);
        col.addView(historyList);

        // ---- details (collapsible) ----
        LinearLayout detCard = card();
        LinearLayout detHead = new LinearLayout(this);
        detHead.setGravity(Gravity.CENTER_VERTICAL);
        TextView dt = text("Details · benchmark & device", 15, C_TEXT, true);
        ImageView arrow = new ImageView(this);
        arrow.setImageDrawable(icon(R.drawable.ic_expand, C_SUB));
        detHead.addView(dt, new LinearLayout.LayoutParams(0, -2, 1));
        detHead.addView(arrow, new LinearLayout.LayoutParams(dp(24), dp(24)));
        detailsBody = new LinearLayout(this);
        detailsBody.setOrientation(LinearLayout.VERTICAL);
        detailsBody.setVisibility(View.GONE);
        LinearLayout detBtns = new LinearLayout(this);
        TextView benchBtn = pill("Test speed (2, 4, 6, 8 threads)", R.drawable.ic_speed);
        useFastestBtn = pill("Use fastest", R.drawable.ic_speed);
        useFastestBtn.setVisibility(View.GONE);
        useFastestBtn.setOnClickListener(v -> {
            if (fastestThreads > 0) { prefs.edit().putInt("threads", fastestThreads).apply(); buildThreadRow();
                Toast.makeText(this, "Threads set to " + fastestThreads, Toast.LENGTH_SHORT).show(); }
        });
        TextView copyBtn = pill("Copy", R.drawable.ic_copy);
        detBtns.addView(benchBtn, pl); detBtns.addView(copyBtn, new LinearLayout.LayoutParams(pl));
        benchOut = text("", 11, C_TEXT, false);
        benchOut.setTypeface(Typeface.MONOSPACE);
        benchOut.setTextIsSelectable(true);
        HorizontalScrollView bs = new HorizontalScrollView(this);
        bs.addView(benchOut);
        LinearLayout.LayoutParams dbl = new LinearLayout.LayoutParams(-1, -2); dbl.topMargin = dp(10);
        detailsBody.addView(text("Runs your last recording 4 times to find the fastest thread count. The report is also saved as benchmark-*.txt next to the model.", 12, C_SUB, false), dbl);
        detailsBody.addView(detBtns, new LinearLayout.LayoutParams(dbl));
        detailsBody.addView(useFastestBtn, new LinearLayout.LayoutParams(-2, -2));
        detailsBody.addView(bs, new LinearLayout.LayoutParams(dbl));
        detCard.addView(detHead); detCard.addView(detailsBody);
        LinearLayout.LayoutParams dl = cardLp(); dl.topMargin = dp(12);
        col.addView(detCard, dl);
        detHead.setOnClickListener(v -> {
            boolean open = detailsBody.getVisibility() != View.VISIBLE;
            detailsBody.setVisibility(open ? View.VISIBLE : View.GONE);
            arrow.setImageDrawable(icon(open ? R.drawable.ic_collapse : R.drawable.ic_expand, C_SUB));
        });

        // ---- listeners ----
        recBtn.setOnClickListener(v -> { if (recording) { stopPressedAt = Latency.now(); recording = false; } else startRecording(); });
        wavBtn.setOnClickListener(v -> pick("audio/*", REQ_WAV));
        modelBtn.setOnClickListener(v -> pick("*/*", REQ_MODEL));
        benchBtn.setOnClickListener(v -> benchmark());
        copyBtn.setOnClickListener(v -> {
            android.content.ClipboardManager cm = (android.content.ClipboardManager) getSystemService(CLIPBOARD_SERVICE);
            cm.setPrimaryClip(android.content.ClipData.newPlainText("ASR benchmark", benchOut.getText()));
            Toast.makeText(this, "Copied", Toast.LENGTH_SHORT).show();
        });

        ScrollView sv = new ScrollView(this);
        sv.setBackgroundColor(C_BG);
        sv.setFillViewport(true);
        sv.setClipToPadding(false);
        sv.addView(col);
        // Android 15+ draws edge-to-edge: keep content clear of status/navigation bars and cutouts.
        sv.setOnApplyWindowInsetsListener((v, in) -> {
            int t, bt, l, r;
            if (Build.VERSION.SDK_INT >= 30) {
                android.graphics.Insets i = in.getInsets(WindowInsets.Type.systemBars() | WindowInsets.Type.displayCutout());
                t = i.top; bt = i.bottom; l = i.left; r = i.right;
            } else { t = in.getSystemWindowInsetTop(); bt = in.getSystemWindowInsetBottom(); l = in.getSystemWindowInsetLeft(); r = in.getSystemWindowInsetRight(); }
            v.setPadding(l, t, r, bt);
            return in;
        });
        return sv;
    }

    /** Thread count used for transcription: saved choice (capped to cores) or Auto. */
    private int threads() {
        int t = prefs.getInt("threads", 0);
        int cores = Runtime.getRuntime().availableProcessors();
        return t <= 0 ? defaultThreads : Math.max(1, Math.min(t, cores));
    }

    private void buildThreadRow() {
        threadRow.removeAllViews();
        int cores = Runtime.getRuntime().availableProcessors();
        int sel = prefs.getInt("threads", 0);
        java.util.LinkedHashMap<Integer, String> opts = new java.util.LinkedHashMap<>();
        opts.put(0, "Auto (" + defaultThreads + ")");
        for (int t : new int[]{2, 4, 6, 8}) if (t <= cores) opts.put(t, String.valueOf(t));
        for (java.util.Map.Entry<Integer, String> e : opts.entrySet()) {
            final int val = e.getKey();
            boolean on = val == sel;
            TextView c = chip(e.getValue(), on ? C_PRIMARY : C_PRIMARY_SOFT, on ? Color.WHITE : C_TEXT);
            c.setPadding(dp(14), dp(7), dp(14), dp(7));
            c.setClickable(true);
            c.setOnClickListener(v -> { prefs.edit().putInt("threads", val).apply(); buildThreadRow(); });
            LinearLayout.LayoutParams l = wrap(); l.setMargins(0, dp(2), dp(6), dp(2));
            threadRow.addView(c, l);
        }
    }

    // ---- small UI factories ----
    private TextView text(String s, float sp, int color, boolean bold) {
        TextView t = new TextView(this);
        t.setText(s); t.setTextSize(sp); t.setTextColor(color);
        t.setTypeface(Typeface.create(bold ? "sans-serif-medium" : "sans-serif", Typeface.NORMAL));
        return t;
    }

    private LinearLayout card() {
        LinearLayout c = new LinearLayout(this);
        c.setOrientation(LinearLayout.VERTICAL);
        GradientDrawable g = new GradientDrawable();
        g.setColor(C_CARD); g.setCornerRadius(dp(20));
        c.setBackground(g);
        c.setElevation(dp(2));
        c.setPadding(dp(16), dp(14), dp(16), dp(14));
        return c;
    }

    private LinearLayout.LayoutParams cardLp() {
        LinearLayout.LayoutParams l = new LinearLayout.LayoutParams(-1, -2);
        l.setMargins(dp(2), dp(6), dp(2), dp(6));
        return l;
    }

    private static LinearLayout.LayoutParams wrap() { return new LinearLayout.LayoutParams(-2, -2); }

    private View divider() {
        View v = new View(this);
        v.setBackgroundColor(C_LINE);
        LinearLayout.LayoutParams l = new LinearLayout.LayoutParams(-1, Math.max(1, dp(1) / 2 + 1));
        l.setMargins(0, dp(12), 0, dp(12));
        v.setLayoutParams(l);
        return v;
    }

    private static GradientDrawable oval(int color) {
        GradientDrawable g = new GradientDrawable();
        g.setShape(GradientDrawable.OVAL); g.setColor(color);
        return g;
    }

    private Drawable ripple(Drawable content, boolean oval) {
        Drawable mask = oval ? oval(Color.WHITE) : content;
        return new RippleDrawable(ColorStateList.valueOf(0x33FFFFFF), content, mask);
    }

    private Drawable icon(int res, int color) {
        Drawable d = getDrawable(res).mutate();
        d.setTint(color);
        return d;
    }

    private TextView chip(String s, int bg, int fg) {
        TextView t = text(s, 12, fg, true);
        GradientDrawable g = new GradientDrawable();
        g.setColor(bg); g.setCornerRadius(dp(14));
        t.setBackground(g);
        t.setPadding(dp(10), dp(4), dp(10), dp(4));
        return t;
    }

    private TextView statChip(String label, String value) {
        TextView t = chip("", C_PRIMARY_SOFT, C_TEXT);
        android.text.SpannableStringBuilder sb = new android.text.SpannableStringBuilder();
        sb.append(label).append("  ");
        sb.setSpan(new android.text.style.ForegroundColorSpan(C_SUB), 0, sb.length(), 0);
        int s = sb.length();
        sb.append(value);
        sb.setSpan(new android.text.style.StyleSpan(Typeface.BOLD), s, sb.length(), 0);
        t.setText(sb);
        t.setTypeface(Typeface.create("sans-serif", Typeface.NORMAL));
        LinearLayout.LayoutParams l = wrap(); l.setMargins(0, dp(3), dp(6), dp(3));
        t.setLayoutParams(l);
        return t;
    }

    private TextView pill(String s, int iconRes) {
        TextView t = text(s, 14, C_PRIMARY, true);
        GradientDrawable g = new GradientDrawable();
        g.setColor(Color.TRANSPARENT); g.setCornerRadius(dp(22)); g.setStroke(dp(1) + 1, 0xFF9FB4C8);
        t.setBackground(new RippleDrawable(ColorStateList.valueOf(0x2256789A), g, null));
        Drawable ic = icon(iconRes, C_PRIMARY);
        ic.setBounds(0, 0, dp(18), dp(18));
        t.setCompoundDrawables(ic, null, null, null);
        t.setCompoundDrawablePadding(dp(6));
        t.setGravity(Gravity.CENTER_VERTICAL);
        t.setPadding(dp(14), dp(9), dp(16), dp(9));
        t.setClickable(true);
        return t;
    }

    private void setModelChip(String s, int dotColor) {
        ui.post(() -> {
            android.text.SpannableStringBuilder sb = new android.text.SpannableStringBuilder("● ");
            sb.setSpan(new android.text.style.ForegroundColorSpan(dotColor), 0, 1, 0);
            sb.append(s);
            modelChip.setText(sb);
        });
    }

    private void setRecordingUi(boolean on) {
        recBtn.setBackground(ripple(oval(on ? C_REC : C_PRIMARY), true));
        recIcon.setImageDrawable(icon(on ? R.drawable.ic_stop : R.drawable.ic_mic, Color.WHITE));
        if (on) {
            recStartMs = System.currentTimeMillis();
            timerText.setText("00:00");
            ui.post(timerTick);
            wave.startLive();
            pulseAnim = ObjectAnimator.ofPropertyValuesHolder(pulse,
                    PropertyValuesHolder.ofFloat("scaleX", 1f, 1.45f),
                    PropertyValuesHolder.ofFloat("scaleY", 1f, 1.45f),
                    PropertyValuesHolder.ofFloat("alpha", 0.35f, 0f));
            pulseAnim.setDuration(1200);
            pulseAnim.setRepeatCount(ValueAnimator.INFINITE);
            pulseAnim.start();
        } else {
            if (pulseAnim != null) pulseAnim.cancel();
            pulse.setAlpha(0f);
            wave.stopLive();
        }
    }

    private void showResult(String text, String label, double audio, double took, int threads, double[] tt, Latency lat) {
        final double postedAt = Latency.now();
        ui.post(() -> {
            if (resultText.getTag() != null) addHistory((String) resultText.getTag(), (String) resultLabel.getTag());
            String shown = text.isEmpty() ? "(no speech recognized)" : text;
            resultText.setText(shown);
            resultText.setTextColor(text.isEmpty() ? C_SUB : C_TEXT);
            resultText.setTag(shown);
            String time = new java.text.SimpleDateFormat("HH:mm", Locale.getDefault()).format(new java.util.Date());
            String meta = String.format(Locale.ROOT, "%s · %s · audio %.1f s · took %.1f s", time, label, audio, took);
            resultLabel.setText("live".equals(label) ? "RESULT · LIVE" : "RESULT");
            resultLabel.setTag(meta);
            statsRow.removeAllViews();
            statsRow.addView(statChip("Audio", String.format(Locale.ROOT, "%.1f s", audio)));
            statsRow.addView(statChip("Total", String.format(Locale.ROOT, "%.1f s", took)));
            statsRow.addView(statChip("Speed", speed(took, audio)));
            statsRow.addView(statChip("Threads", String.valueOf(threads)));
            if (tt != null) {
                double enc = tt[2] / 1000, dec = tt[3] / 1000;
                statsRow.addView(statChip("Listening", String.format(Locale.ROOT, "%.1f s", enc)));
                statsRow.addView(statChip("Writing", String.format(Locale.ROOT, "%.1f s · %d tok", dec, (int) tt[5])));
            }
            statsRow.addView(statChip("Model load", String.format(Locale.ROOT, "%.1f s", lastLoadSec)));
            if (lat != null) {
                lat.uiMs = Latency.now() - postedAt;
                lat.endAt = Latency.now();
                lastLatencyReport = lat.table();
                latencyOut.setText(lastLatencyReport);
                statsRow.addView(statChip("Stop → text", String.format(Locale.ROOT, "%.2f s", lat.e2e() / 1000)), 2);
            }
        });
    }

    private void addHistory(String text, String meta) {
        if (historyList.getChildCount() == 1 && historyList.getChildAt(0).getTag() == null
                && historyList.getChildAt(0) instanceof TextView && "Nothing yet.".contentEquals(((TextView) historyList.getChildAt(0)).getText()))
            historyList.removeAllViews();
        LinearLayout c = card();
        c.setTag("h");
        c.setElevation(dp(1));
        TextView m = text(meta == null ? "" : meta, 11, C_SUB, false);
        TextView t = text(text, 16, C_TEXT, false);
        t.setLineSpacing(0, 1.25f);
        t.setTextIsSelectable(true);
        t.setPadding(0, dp(4), 0, 0);
        c.addView(m); c.addView(t);
        LinearLayout.LayoutParams l = new LinearLayout.LayoutParams(-1, -2);
        l.setMargins(dp(2), dp(4), dp(2), dp(4));
        historyList.addView(c, 0, l);
    }

    // =====================================================================================
    // Model
    // =====================================================================================
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
        File[] fs = dir.listFiles((d, n) -> n.toLowerCase(Locale.ROOT).endsWith(".bin") && (!ggmlOnly || n.toLowerCase(Locale.ROOT).contains("ggml")));
        if (fs == null || fs.length == 0) return null;
        Arrays.sort(fs);
        return fs[0];
    }

    /** Worker thread only. */
    private boolean ensureModel() {
        File m = findModel();
        if (m == null) { setModelChip("No model", C_WARN); setStatus("No model file in " + modelDir()); return false; }
        lastEnsureLoadMs = -1;
        if (ctx != 0 && m.getAbsolutePath().equals(loadedModel)) return true;
        if (ctx != 0) { Whisper.free(ctx); ctx = 0; }
        String desc = m.getName() + " · " + (m.length() >> 20) + " MB";
        setModelChip(desc + " · loading…", C_WARN);
        setStatus("Loading model… (first time takes a while)");
        long t0 = System.currentTimeMillis();
        ctx = Whisper.init(m.getAbsolutePath());
        if (ctx == 0) { setModelChip(desc + " · failed", C_REC); toastStatus("Failed to load model " + m.getName() + " (not enough RAM or bad file)."); return false; }
        loadedModel = m.getAbsolutePath();
        lastLoadSec = (System.currentTimeMillis() - t0) / 1000.0;
        lastEnsureLoadMs = lastLoadSec * 1000;
        setModelChip(desc + " · ready", C_OK);
        return true;
    }

    // =====================================================================================
    // Recording (+ live mode)
    // =====================================================================================
    private void startRecording() {
        if (busy) { toastStatus("Busy - wait until the current task finishes."); return; }
        if (Build.VERSION.SDK_INT >= 23 && checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQ_MIC);
            return;
        }
        final boolean live = liveSwitch.isChecked();
        synchronized (audioLock) { recLen = 0; }
        liveLats.clear();
        committed.setLength(0); segQueued = 0; segDone = 0; lastDoneEnd = 0; liveInferSec = 0;
        recording = true;
        liveSwitch.setEnabled(false);
        setRecordingUi(true);
        setStatus(live ? "Speak in sentences; text appears after each pause" : "Recording… tap again to stop");
        liveInfo.setVisibility(live ? View.VISIBLE : View.GONE);
        liveInfo.setText("Listening…"); liveInfo.setTextColor(C_SUB);
        resultLabel.setText(live ? "RESULT · LIVE" : "RESULT");
        if (live) { if (resultText.getTag() != null) addHistory((String) resultText.getTag(), (String) resultLabel.getTag());
            resultText.setText(""); resultText.setTextColor(C_TEXT); resultText.setTag(null); statsRow.removeAllViews(); worker.execute(this::ensureModel); }

        new Thread(() -> {
            int minBuf = AudioRecord.getMinBufferSize(SR, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
            AudioRecord ar = new AudioRecord(MediaRecorder.AudioSource.MIC, SR,
                    AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT, Math.max(minBuf, SR * 2));
            if (ar.getState() != AudioRecord.STATE_INITIALIZED) {
                recording = false; ar.release();
                ui.post(() -> { setRecordingUi(false); liveSwitch.setEnabled(true); });
                toastStatus("Microphone not available.");
                return;
            }
            Vad vad = live ? new Vad() : null;
            short[] buf = new short[LEVEL_WIN];
            ar.startRecording();
            while (recording) {
                int n = ar.read(buf, 0, buf.length);   // blocks until a full 30 ms frame (or error)
                if (n <= 0) continue;
                double s = 0;
                synchronized (audioLock) {
                    if (recLen + n > rec.length) rec = Arrays.copyOf(rec, Math.max(rec.length * 2, recLen + n));
                    System.arraycopy(buf, 0, rec, recLen, n);
                    recLen += n;
                    if (recLen >= SR * MAX_SECONDS) recording = false;
                }
                for (int i = 0; i < n; i++) { float v = buf[i] / 32768f; s += v * v; }
                float level = (float) Math.sqrt(s / n);
                wave.addLevel(level);
                if (vad != null) { int[] seg = vad.push(level); if (seg != null) enqueueSegment(seg[0], seg[1]); updateLiveInfo(); }
            }
            ar.stop(); ar.release();
            final float[] all;
            final int endSamples;
            synchronized (audioLock) { all = toFloat(rec, 0, recLen); endSamples = recLen; }
            final Latency micLat = new Latency();
            micLat.source = "mic"; micLat.audioName = "recording";
            micLat.startAt = stopPressedAt > 0 ? stopPressedAt : Latency.now();
            micLat.queuedAt = Latency.now();
            micLat.prepMs = micLat.queuedAt - micLat.startAt;
            ui.post(() -> { setRecordingUi(false); liveSwitch.setEnabled(true); });
            wave.setAudio(all);
            if (live) {
                int[] last = vad.flush(endSamples);
                if (last != null) enqueueSegment(last[0], last[1]);
                finishLive(all);
            } else transcribe(all, "mic", micLat);
        }).start();
    }

    private static float[] toFloat(short[] a, int from, int to) {
        float[] f = new float[Math.max(0, to - from)];
        for (int i = 0; i < f.length; i++) f[i] = a[from + i] / 32768f;
        return f;
    }

    /**
     * Energy-based voice activity detector working on 30 ms frames.
     * - Noise floor: average of the first ~0.5 s, then slowly follows quiet frames.
     * - Speech starts after 3 loud frames in a row; ends after >= 0.8 s of quiet (hangover).
     * - Segments shorter than 1.5 s are not cut (they merge with what follows).
     * - Segments longer than 25 s are cut at the quietest 200 ms inside their last 3 s.
     * - 0.3 s of padding is kept before and after each segment.
     * Returns {startSample, endSample} when a segment is finished.
     */
    private final class Vad {
        static final int F = LEVEL_WIN;                       // samples per frame
        static final int PAD = 10, PAUSE = 27, MIN = 50, MAX = 833, QUIET_WIN = 7, SEARCH = 100, START_RUN = 3;
        float[] lv = new float[SR * 60 / F];
        int n = 0;                                            // frames seen
        float floor = -1; double floorSum = 0;
        boolean speech = false;
        int segStartF = 0, lastVoiceF = 0, loudRun = 0, lastCutF = 0;

        int[] push(float level) {
            if (n == lv.length) lv = Arrays.copyOf(lv, n * 2);
            lv[n] = level;
            int f = n++;
            if (f < 17) { floorSum += level; floor = (float) (floorSum / (f + 1)); return null; } // first ~0.5 s
            float thr = Math.max(0.006f, floor * 3f);
            boolean loud = level > thr;
            if (!loud && level < floor * 2f) floor = floor * 0.98f + level * 0.02f;    // follow the background
            if (!speech) {
                loudRun = loud ? loudRun + 1 : 0;
                if (loudRun >= START_RUN) {
                    speech = true;
                    segStartF = Math.max(lastCutF, f - START_RUN + 1 - PAD);
                    lastVoiceF = f;
                }
                return null;
            }
            if (loud) lastVoiceF = f;
            int len = f - segStartF;
            if (f - lastVoiceF >= PAUSE && lastVoiceF - segStartF >= MIN) {   // real pause after enough speech
                int end = Math.min(f, lastVoiceF + PAD);
                speech = false; loudRun = 0; lastCutF = end;
                return new int[]{segStartF * F, end * F};
            }
            if (len >= MAX) {                                                  // too long: cut at the quietest spot
                int best = f - QUIET_WIN; double bestE = Double.MAX_VALUE;
                for (int i = Math.max(segStartF + MIN, f - SEARCH); i + QUIET_WIN <= f; i++) {
                    double e = 0; for (int k = 0; k < QUIET_WIN; k++) e += lv[i + k];
                    if (e < bestE) { bestE = e; best = i; }
                }
                int cut = best + QUIET_WIN / 2;
                int[] seg = {segStartF * F, cut * F};
                segStartF = cut; lastCutF = cut;
                return seg;
            }
            return null;
        }

        /** On Stop: whatever speech is still open becomes the last segment. */
        int[] flush(int endSamples) {
            if (!speech) return null;
            speech = false;
            int start = segStartF * F;
            return endSamples - start > SR / 4 ? new int[]{start, endSamples} : null;
        }
    }

    /** Recorder thread: queue a finished segment. The single-thread worker runs them FIFO, one at a time. */
    private void enqueueSegment(final int start, final int end) {
        final int no = ++segQueued;
        final Latency lat = new Latency();
        lat.source = "live"; lat.segNo = no; lat.audioSec = (end - start) / (double) SR;
        lat.startAt = Latency.now(); lat.queuedAt = lat.startAt;
        if (no == 1) liveLats.clear();
        updateLiveInfo();
        worker.execute(() -> {
            try {
                lat.workerStartAt = Latency.now();
                if (!ensureModel()) return;
                lat.loadMs = lastEnsureLoadMs;
                double p0 = Latency.now();
                float[] seg;
                synchronized (audioLock) { seg = toFloat(rec, start, Math.min(end, recLen)); }
                lat.prepMs = Latency.now() - p0;
                lat.threads = threads();
                String prompt = committed.length() > 0 ? committed.substring(Math.max(0, committed.length() - 200)) : null;
                ui.post(this::updateLiveInfo);
                long t0 = System.currentTimeMillis();
                String t = Whisper.transcribe(ctx, seg, lat.threads, LANG, prompt).trim();
                liveInferSec += (System.currentTimeMillis() - t0) / 1000.0;
                lat.tt = Whisper.timings(ctx);
                lat.inferMs = System.currentTimeMillis() - t0;
                if (!t.isEmpty()) committed.append(committed.length() > 0 ? " " : "").append(t);
                lastDoneEnd = end;
                segDone = no;
                final String shown = committed.toString();
                final double postedAt = Latency.now();
                ui.post(() -> { renderLiveText(shown); updateLiveInfo();
                    lat.uiMs = Latency.now() - postedAt; lat.endAt = Latency.now(); liveLats.add(lat); });
            } catch (Throwable e) {
                segDone = no;
                toastStatus("Live error: " + e);
            }
        });
    }

    private void renderLiveText(String shown) {
        boolean pending = segQueued > segDone;
        android.text.SpannableStringBuilder sb = new android.text.SpannableStringBuilder(shown);
        if (pending) {                                   // subtle placeholder for the segment being processed
            int s = sb.length();
            sb.append(shown.isEmpty() ? "…" : " …");
            sb.setSpan(new android.text.style.ForegroundColorSpan(C_SUB), s, sb.length(), 0);
        }
        resultText.setText(sb);
    }

    /** "Listening…" / "Transcribing segment N…" / amber "Behind by X s". Any thread. */
    private void updateLiveInfo() {
        final int len;
        synchronized (audioLock) { len = recLen; }
        final int q = segQueued, d = segDone;
        final double behind = q > d ? (len - lastDoneEnd) / (double) SR : 0;
        ui.post(() -> {
            if (liveInfo.getVisibility() != View.VISIBLE) return;
            String s; int col = C_SUB;
            if (q > d && behind > 6) { s = String.format(Locale.ROOT, "Behind by %.0f s · transcribing segment %d of %d…", behind, d + 1, q); col = C_WARN; }
            else if (q > d) s = "Transcribing segment " + (d + 1) + "…" + (ctx == 0 ? " (loading model)" : "");
            else s = recording ? "Listening…" : "Finishing…";
            liveInfo.setText(s); liveInfo.setTextColor(col);
            if (q > d && resultText.getTag() == null) renderLiveText(committed.toString());
        });
    }

    /** After Stop in live mode: queued after all segments (FIFO), then show final result + summed stats. */
    private void finishLive(float[] all) {
        busy = true;
        setStatus("Finishing the remaining segments…");
        updateLiveInfo();
        worker.execute(() -> {
            try {
                lastAudio = all;
                String text = committed.toString();
                double audio = all.length / (double) SR;
                showResult(text, "live", audio, liveInferSec, threads(), null, null);
                final double stopAt = stopPressedAt;
                final String model = loadedModel == null ? "?" : new File(loadedModel).getName();
                final int th = threads();
                ui.post(() -> {   // runs after showResult's UI update
                    java.util.List<Latency> copy;
                    synchronized (liveLats) { copy = new java.util.ArrayList<>(liveLats); }
                    lastLatencyReport = Latency.liveTable(copy, Latency.now() - stopAt, audio, model, th);
                    latencyOut.setText(lastLatencyReport);
                });
                setStatus(String.format(Locale.ROOT, "Done. %d segment(s) transcribed.", segDone));
                ui.post(() -> liveInfo.setVisibility(View.GONE));
            } catch (Throwable t) {
                toastStatus("Error: " + t);
            } finally { busy = false; }
        });
    }

    @Override public void onRequestPermissionsResult(int code, String[] p, int[] r) {
        if (code == REQ_MIC && r.length > 0 && r[0] == PackageManager.PERMISSION_GRANTED) startRecording();
        else if (code == REQ_MIC) toastStatus("Microphone permission denied. Allow it in Settings > Apps.");
    }

    // =====================================================================================
    // Normal transcription
    // =====================================================================================
    private void transcribe(float[] pcm, String label, Latency lat) {
        busy = true;
        worker.execute(() -> {
            try {
                lat.workerStartAt = Latency.now();
                if (!ensureModel()) return;
                lat.loadMs = lastEnsureLoadMs;
                double sec = pcm.length / (double) SR;
                setStatus(String.format(Locale.ROOT, "Transcribing %.1f s of audio…", sec));
                lastAudio = pcm;
                int threads = threads();
                long t0 = System.currentTimeMillis();
                String text = Whisper.transcribe(ctx, pcm, threads, LANG, null).trim();
                double took = (System.currentTimeMillis() - t0) / 1000.0;
                double[] tt = Whisper.timings(ctx);
                lat.tt = tt; lat.inferMs = took * 1000; lat.audioSec = sec; lat.threads = threads;
                lat.model = new File(loadedModel).getName();
                showResult(text, label, sec, took, threads, tt, lat);
                setStatus("Done. Tap the microphone to record again.");
            } catch (Throwable t) {
                toastStatus("Error: " + t);
            } finally { busy = false; }
        });
    }

    // =====================================================================================
    // Files
    // =====================================================================================
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
                setStatus("Choose a file…");
                return;
            } catch (ActivityNotFoundException | SecurityException e) {
                // try next action
            }
        }
        toastStatus("No file picker app found on this phone. Use adb push instead (see README).");
    }

    private String displayName(Uri uri) {
        try (android.database.Cursor c = getContentResolver().query(uri, new String[]{android.provider.OpenableColumns.DISPLAY_NAME}, null, null, null)) {
            if (c != null && c.moveToFirst()) return c.getString(0);
        } catch (Throwable ignored) {}
        String s = uri.getLastPathSegment();
        return s == null ? "file" : s;
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
            final Latency lat = new Latency();
            lat.source = "file"; lat.startAt = Latency.now();
            lat.audioName = displayName(uri);
            worker.execute(() -> {
                try (InputStream in = getContentResolver().openInputStream(uri)) {
                    float[] f = Wav.read16kMono(in);
                    lat.queuedAt = Latency.now();
                    lat.prepMs = lat.queuedAt - lat.startAt;
                    wave.setAudio(f);
                    busy = false;
                    transcribe(f, "file", lat);
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
                        if (mb / 20 != lastMb) { lastMb = mb / 20; setStatus("Copying model… " + mb + (total0 > 0 ? " / " + (total0 >> 20) : "") + " MB"); }
                    }
                    out.close();
                    if (total < (10L << 20)) throw new java.io.IOException("file too small (" + total + " bytes) - not a model");
                    File[] old = modelDir().listFiles((d, nm) -> nm.toLowerCase(Locale.ROOT).endsWith(".bin"));
                    if (old != null) for (File o : old) o.delete();
                    if (ctx != 0) { Whisper.free(ctx); ctx = 0; loadedModel = null; }
                    File dst = new File(modelDir(), "model.bin");
                    if (!tmp.renameTo(dst) && !dst.exists()) throw new java.io.IOException("rename failed");
                    ensureModel();
                    setStatus("Model ready. Tap the microphone to record.");
                } catch (Throwable t) { tmp.delete(); toastStatus("Model copy failed: " + t); }
                finally { busy = false; }
            });
        }
    }

    // =====================================================================================
    // Stats / benchmark
    // =====================================================================================
    static String speed(double took, double audio) {
        if (audio <= 0 || took <= 0) return "?";
        double r = took / audio;
        return r >= 1 ? String.format(Locale.ROOT, "%.2f× slower", r)
                      : String.format(Locale.ROOT, "%.2f× faster", 1 / r);
    }

    private static boolean isEmulatorAbi() { return Build.SUPPORTED_ABIS[0].startsWith("x86"); }

    private String deviceInfo() {
        return "Device: " + Build.MANUFACTURER + " " + Build.MODEL + " | Android " + Build.VERSION.RELEASE + " (API " + Build.VERSION.SDK_INT + ")\n"
                + "ABI: " + Build.SUPPORTED_ABIS[0] + " | cores: " + Runtime.getRuntime().availableProcessors()
                + " | " + CpuInfo.clusters() + " | default threads: " + defaultThreads + "\n"
                + "CPU flags: " + CpuInfo.features() + "\n"
                + "ggml backends: " + Whisper.backendInfo() + "\n"
                + "whisper system info: " + Whisper.systemInfo() + "\n"
                + (isEmulatorAbi() ? "Note: running on emulator/PC CPU (x86) — not phone speed.\n" : "");
    }

    private void benchmark() {
        if (busy || recording) { toastStatus("Busy - wait until the current task finishes."); return; }
        final float[] pcm = lastAudio;
        if (pcm == null) { toastStatus("Record something or transcribe a WAV first; the benchmark re-uses that audio."); return; }
        busy = true;
        worker.execute(() -> {
            try {
                if (!ensureModel()) return;
                int cores = Runtime.getRuntime().availableProcessors();
                java.util.TreeSet<Integer> set = new java.util.TreeSet<>();
                for (int t : new int[]{2, 4, 6, 8}) set.add(Math.max(1, Math.min(t, cores)));
                double sec = pcm.length / (double) SR;
                StringBuilder sb = new StringBuilder();
                sb.append("=== ASR-llm benchmark ").append(new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.ROOT).format(new java.util.Date())).append(" ===\n");
                sb.append(deviceInfo());
                sb.append("Model: ").append(new File(loadedModel).getName()).append(String.format(Locale.ROOT, " | loading model %.1f s | audio %.1f s%n", lastLoadSec, sec));
                if (!lastLatencyReport.isEmpty()) sb.append("\nLast transcription:\n").append(lastLatencyReport).append("\n");
                StringBuilder runs = new StringBuilder();
                sb.append("Threads | Total s | Speed vs real time | Listening (encoder) s | Writing text (decoder) s | Tokens | Tokens/s\n");
                final String head = sb.toString();
                ui.post(() -> benchOut.setText(head));
                String text = "";
                double best = Double.MAX_VALUE; int bestT = 0;
                for (int t : set) {
                    setStatus("Benchmark: " + t + " threads… (each run takes about as long as one transcription)");
                    long t0 = System.currentTimeMillis();
                    Latency rl = new Latency();
                    rl.source = "speed test"; rl.audioName = "last audio"; rl.audioSec = sec; rl.threads = t;
                    rl.model = new File(loadedModel).getName();
                    rl.startAt = Latency.now(); rl.queuedAt = rl.startAt; rl.workerStartAt = rl.startAt;
                    text = Whisper.transcribe(ctx, pcm, t, LANG, null).trim();
                    double took = (System.currentTimeMillis() - t0) / 1000.0;
                    double[] x = Whisper.timings(ctx);
                    rl.tt = x; rl.endAt = Latency.now();
                    runs.append("\n").append(rl.table().replace("7 End-to-end                    ", "7 Total (this run)              "));
                    double dec = x[3] / 1000;
                    if (took < best) { best = took; bestT = t; }
                    sb.append(String.format(Locale.ROOT, "%7d | %7.1f | %-18s | %21.1f | %24.1f | %6d | %8.1f%n",
                            t, took, speed(took, sec), x[2] / 1000, dec, (int) x[5], dec > 0 ? x[5] / dec : 0));
                    final String cur = sb.toString();
                    ui.post(() -> benchOut.setText(cur));
                }
                sb.append("Fastest: ").append(bestT).append(" threads\n");
                sb.append(runs);
                sb.append("Last transcript: ").append(text).append("\n");
                fastestThreads = bestT;
                final int bt = bestT;
                ui.post(() -> { useFastestBtn.setText("Use fastest (" + bt + " threads)"); useFastestBtn.setVisibility(View.VISIBLE); });
                File out = new File(modelDir(), "benchmark-" + System.currentTimeMillis() + ".txt");
                try (java.io.Writer w = new java.io.OutputStreamWriter(new FileOutputStream(out), "UTF-8")) { w.write(sb.toString()); }
                final String fin = sb.toString();
                ui.post(() -> benchOut.setText(fin));
                setStatus("Benchmark done. Saved to " + out.getAbsolutePath());
            } catch (Throwable t) {
                toastStatus("Benchmark error: " + t);
            } finally { busy = false; }
        });
    }

    private void setStatus(String s) { ui.post(() -> status.setText(s)); }
    private int dp(int v) { return (int) (v * getResources().getDisplayMetrics().density + 0.5f); }

    @Override protected void onDestroy() {
        recording = false;
        worker.execute(() -> { if (ctx != 0) { Whisper.free(ctx); ctx = 0; } });
        worker.shutdown();
        super.onDestroy();
    }
}
