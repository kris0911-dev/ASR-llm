package com.asrllm;

import android.content.Context;
import android.graphics.Canvas;
import android.graphics.Paint;
import android.graphics.RectF;
import android.view.View;

/**
 * Voice graph. While recording: bars scroll in from the right (one bar per ~30 ms RMS level).
 * After stop / WAV load: shows the whole recording squeezed to the view width.
 */
public class WaveformView extends View {
    private static final int MAX_LEVELS = 16000 * 60 * 10 / 480; // 10 min of 30 ms levels
    private final float[] levels = new float[MAX_LEVELS];
    private int count = 0;
    private boolean live = false;
    private float gain = 1f;
    private final Paint bar = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final Paint mid = new Paint(Paint.ANTI_ALIAS_FLAG);
    private final RectF r = new RectF();
    private final float d;
    private int liveColor = 0xFFD9625B, idleColor = 0xFF5B7C99;
    private final Runnable frame = new Runnable() {
        @Override public void run() { invalidate(); if (live) postOnAnimation(this); }
    };

    public WaveformView(Context c) {
        super(c);
        d = c.getResources().getDisplayMetrics().density;
        mid.setColor(0x335B7C99); mid.setStrokeWidth(d);
    }

    void setColors(int liveC, int idleC) { liveColor = liveC; idleColor = idleC; }

    /** Clear and enter live (scrolling) mode. UI thread. */
    void startLive() {
        synchronized (levels) { count = 0; gain = 1f; }
        live = true;
        removeCallbacks(frame);
        postOnAnimation(frame);
    }

    /** Leave live mode; show whole recording. UI thread. */
    void stopLive() { live = false; invalidate(); }

    /** Append one RMS level (0..1). Any thread. */
    void addLevel(float v) {
        synchronized (levels) { if (count < MAX_LEVELS) levels[count++] = v; }
    }

    /** Show a full waveform of the given 16 kHz audio. Any thread. */
    void setAudio(float[] pcm) {
        synchronized (levels) {
            count = 0;
            for (int i = 0; i + 480 <= pcm.length && count < MAX_LEVELS; i += 480) {
                double s = 0;
                for (int j = 0; j < 480; j++) s += pcm[i + j] * pcm[i + j];
                levels[count++] = (float) Math.sqrt(s / 480);
            }
        }
        live = false;
        postInvalidate();
    }

    @Override protected void onMeasure(int w, int h) {
        setMeasuredDimension(MeasureSpec.getSize(w), (int) (96 * d));
    }

    @Override protected void onDraw(Canvas c) {
        int w = getWidth(), h = getHeight();
        float cy = h / 2f;
        c.drawLine(0, cy, w, cy, mid);
        float barW = 3 * d, gap = 2 * d, step = barW + gap;
        int slots = (int) (w / step);
        bar.setColor(live ? liveColor : idleColor);
        synchronized (levels) {
            if (count == 0) return;
            // auto-gain so quiet mics are still visible (peak of shown data -> ~90% height)
            float peak = 0.02f;
            int from = live ? Math.max(0, count - slots) : 0;
            for (int i = from; i < count; i++) peak = Math.max(peak, levels[i]);
            float target = 0.9f / peak;
            gain = live ? gain + (target - gain) * 0.1f : target;
            if (live) {
                int n = Math.min(slots, count);
                for (int k = 0; k < n; k++) {
                    float v = Math.min(1f, levels[count - n + k] * gain);
                    float x = w - (n - k) * step;
                    drawBar(c, x, cy, barW, v, h);
                }
            } else {
                int n = Math.min(slots, count);
                for (int k = 0; k < n; k++) {
                    int a = (int) ((long) k * count / n), b = Math.max(a + 1, (int) ((long) (k + 1) * count / n));
                    float m = 0;
                    for (int i = a; i < b && i < count; i++) m = Math.max(m, levels[i]);
                    drawBar(c, k * step, cy, barW, Math.min(1f, m * gain), h);
                }
            }
        }
    }

    private void drawBar(Canvas c, float x, float cy, float bw, float v, int h) {
        float half = Math.max(d, v * (h / 2f - 2 * d));
        r.set(x, cy - half, x + bw, cy + half);
        c.drawRoundRect(r, bw / 2, bw / 2, bar);
    }
}
