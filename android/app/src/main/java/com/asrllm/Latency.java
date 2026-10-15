package com.asrllm;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/** Timing record for one transcription (or one live segment) and plain-language tables. All times in ms. */
final class Latency {
    static double now() { return System.nanoTime() / 1e6; }

    String source = "", audioName = "", model = "";
    int threads, segNo;
    double audioSec;
    double startAt;        // e2e start: Stop pressed / file chosen / segment detected
    double prepMs;         // read WAV / copy mic buffer
    double queuedAt, workerStartAt;
    double loadMs = -1;    // -1 = model was already loaded
    double inferMs;        // Java wall time around Whisper.transcribe
    double[] tt;           // Whisper.timings()
    double uiMs;           // posting result to the screen + drawing text
    double endAt;          // text shown

    double queueMs() { return Math.max(0, workerStartAt - queuedAt); }
    double e2e() { return endAt - startAt; }
    double mel() { return tt[1]; }
    double enc() { return tt[2]; }
    double dec() { return tt[3]; }
    double samp() { return tt[4]; }
    int tokens() { return (int) tt[5]; }
    int windows() { return Math.max(1, (int) tt[6]); }
    double post() { return tt[13]; }

    private static String row(String label, double ms, double total, String note) {
        return String.format(Locale.ROOT, "%-34s %9.0f %8.2f %6s  %s%n", label, ms, ms / 1000,
                total > 0 ? String.format(Locale.ROOT, "%5.1f", 100 * ms / total) : "", note == null ? "" : note);
    }

    private static String info(String label, String note) {
        return String.format(Locale.ROOT, "%-34s %9s %8s %6s  %s%n", label, "", "", "", note);
    }

    private static String head() {
        return String.format(Locale.ROOT, "%-34s %9s %8s %6s  %s%n", "Step", "ms", "s", "%", "notes")
                + "---------------------------------------------------------------------------------\n";
    }

    String title() {
        return String.format(Locale.ROOT, "Latency · %s%s · audio %.1f s · %s · %d threads%n", source,
                audioName.isEmpty() ? "" : " (" + audioName + ")", audioSec, model, threads);
    }

    /** Full breakdown for a normal (non-live) transcription. Rows add up to end-to-end. */
    String table() {
        double total = e2e();
        double load = Math.max(0, loadMs);
        double named = prepMs + queueMs() + load + mel() + enc() + dec() + samp() + post() + uiMs;
        double other = total - named;
        StringBuilder sb = new StringBuilder(title()).append(head());
        sb.append(row("1 Read/prepare audio", prepMs, total, source.equals("mic") ? "stop mic + copy buffer" : "decode WAV + resample"));
        sb.append(row("  Waiting in queue", queueMs(), total, null));
        sb.append(row("2 Audio -> spectrogram (mel)", mel(), total, null));
        sb.append(row("3 Listening (encoder)", enc(), total, String.format(Locale.ROOT, "%d window(s) x %.0f ms", windows(), enc() / windows())));
        sb.append(row("4 Writing text (decoder)", dec(), total, String.format(Locale.ROOT, "%d tokens, %.0f ms/token, %.1f tokens/s",
                tokens(), tokens() > 0 ? dec() / tokens() : 0, dec() > 0 ? tokens() * 1000.0 / dec() : 0)));
        sb.append(info("    prompt / batch / single-token", String.format(Locale.ROOT, "%.0f / %.0f / %.0f ms (%d batch, %d single calls)",
                tt[9], tt[8], tt[7], (int) tt[11], (int) tt[10])));
        sb.append(row("    first token ready", tt[12] < 0 ? 0 : tt[12], 0, tt[12] < 0 ? "no tokens" : "ms after transcription start"));
        sb.append(row("  Choosing tokens (sampling)", samp(), total, null));
        sb.append(row("5 Text assembly", post(), total, null));
        sb.append(row("  Screen update", uiMs, total, null));
        sb.append(row("6 Loading model", load, total, loadMs < 0 ? "already loaded" : "first use"));
        sb.append(row("  Other / unaccounted", other, total, "threads start/stop, JNI, scheduling"));
        sb.append("---------------------------------------------------------------------------------\n");
        sb.append(row("7 End-to-end", total, total, endToEndNote()));
        return sb.toString();
    }

    private String endToEndNote() {
        return source.equals("mic") ? "Stop pressed -> text shown" : source.equals("file") ? "file chosen -> text shown" : "";
    }

    /** Live mode: one line per segment + summed stages + Stop -> final text. */
    static String liveTable(List<Latency> segs, double stopToShownMs, double audioSec, String model, int threads) {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(Locale.ROOT, "Latency · live · audio %.1f s · %s · %d threads · %d segment(s)%n", audioSec, model, threads, segs.size()));
        sb.append(String.format(Locale.ROOT, "%3s %7s %7s %6s %6s %8s %8s %6s %8s %7s %8s%n",
                "seg", "audio s", "wait", "prep", "mel", "encoder", "decoder", "tokens", "1st tok", "screen", "total"));
        double sPrep = 0, sWait = 0, sMel = 0, sEnc = 0, sDec = 0, sSamp = 0, sPost = 0, sUi = 0, sLoad = 0; int sTok = 0;
        for (Latency l : segs) {
            sb.append(String.format(Locale.ROOT, "%3d %7.1f %7.0f %6.0f %6.0f %8.0f %8.0f %6d %8.0f %7.0f %8.0f%n",
                    l.segNo, l.audioSec, l.queueMs(), l.prepMs, l.mel(), l.enc(), l.dec(), l.tokens(), l.tt[12], l.uiMs, l.e2e()));
            sPrep += l.prepMs; sWait += l.queueMs(); sMel += l.mel(); sEnc += l.enc(); sDec += l.dec(); sSamp += l.samp();
            sPost += l.post(); sUi += l.uiMs; sTok += l.tokens(); sLoad += Math.max(0, l.loadMs);
        }
        sb.append("(times in ms; total = segment detected -> its text shown, includes waiting)\n\n");
        sb.append("Summed over segments:\n").append(head());
        sb.append(row("1 Read/prepare audio", sPrep, 0, "copy segment from mic buffer"));
        sb.append(row("  Waiting in queue", sWait, 0, null));
        sb.append(row("2 Audio -> spectrogram (mel)", sMel, 0, null));
        sb.append(row("3 Listening (encoder)", sEnc, 0, null));
        sb.append(row("4 Writing text (decoder)", sDec, 0, String.format(Locale.ROOT, "%d tokens, %.0f ms/token", sTok, sTok > 0 ? sDec / sTok : 0)));
        sb.append(row("  Choosing tokens (sampling)", sSamp, 0, null));
        sb.append(row("5 Text assembly + screen update", sPost + sUi, 0, null));
        sb.append(row("6 Loading model", sLoad, 0, sLoad > 0 ? "first use" : "already loaded"));
        sb.append(row("7 End-to-end (Stop -> final text)", stopToShownMs, 0, "most work ran while you spoke, so sums need not match"));
        return sb.toString();
    }

    static List<Latency> newList() { return new ArrayList<>(); }
}
