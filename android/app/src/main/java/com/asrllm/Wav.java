package com.asrllm;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Reads a WAV (PCM 8/16/24/32-bit or float32, any rate, any channels) into 16 kHz mono floats. */
final class Wav {
    static float[] read16kMono(InputStream in) throws IOException {
        ByteArrayOutputStream bo = new ByteArrayOutputStream();
        byte[] b = new byte[65536]; int n;
        while ((n = in.read(b)) > 0) bo.write(b, 0, n);
        ByteBuffer bb = ByteBuffer.wrap(bo.toByteArray()).order(ByteOrder.LITTLE_ENDIAN);
        if (bb.remaining() < 12 || bb.getInt(0) != 0x46464952 || bb.getInt(8) != 0x45564157)
            throw new IOException("not a RIFF/WAVE file (only .wav is supported)");
        int fmt = 0, ch = 0, rate = 0, bits = 0, dataOff = -1, dataLen = 0, p = 12;
        while (p + 8 <= bb.limit()) {
            int id = bb.getInt(p), len = bb.getInt(p + 4);
            if (id == 0x20746d66) { // "fmt "
                fmt = bb.getShort(p + 8) & 0xffff; ch = bb.getShort(p + 10); rate = bb.getInt(p + 12); bits = bb.getShort(p + 22);
                if (fmt == 0xFFFE) fmt = bb.getShort(p + 32) & 0xffff; // WAVE_FORMAT_EXTENSIBLE
            } else if (id == 0x61746164) { // "data"
                dataOff = p + 8; dataLen = Math.min(len < 0 ? Integer.MAX_VALUE : len, bb.limit() - dataOff); break;
            }
            p += 8 + len + (len & 1);
        }
        if (dataOff < 0 || ch <= 0 || rate <= 0) throw new IOException("bad WAV header");
        int bps = bits / 8, frames = dataLen / (bps * ch);
        float[] mono = new float[frames];
        for (int i = 0; i < frames; i++) {
            float s = 0;
            for (int c = 0; c < ch; c++) {
                int o = dataOff + (i * ch + c) * bps;
                float v;
                if (fmt == 3 && bits == 32) v = bb.getFloat(o);
                else if (bits == 16) v = bb.getShort(o) / 32768f;
                else if (bits == 8) v = ((bb.get(o) & 0xff) - 128) / 128f;
                else if (bits == 24) v = (((bb.get(o) & 0xff) | ((bb.get(o + 1) & 0xff) << 8) | (bb.get(o + 2) << 16))) / 8388608f;
                else if (bits == 32) v = bb.getInt(o) / 2147483648f;
                else throw new IOException("unsupported WAV sample format (" + bits + " bit)");
                s += v;
            }
            mono[i] = s / ch;
        }
        if (rate == 16000) return mono;
        int outN = (int) ((long) frames * 16000 / rate);   // simple linear resampling
        float[] out = new float[outN];
        double step = rate / 16000.0;
        for (int i = 0; i < outN; i++) {
            double x = i * step; int i0 = (int) x; int i1 = Math.min(i0 + 1, frames - 1);
            double f = x - i0;
            out[i] = (float) (mono[i0] * (1 - f) + mono[i1] * f);
        }
        return out;
    }
}
