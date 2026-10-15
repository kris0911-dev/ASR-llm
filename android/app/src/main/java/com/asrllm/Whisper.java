package com.asrllm;

/** Thin JNI wrapper around whisper.cpp. Call from one background thread only. */
final class Whisper {
    static { System.loadLibrary("asrllm"); }
    static native long init(String modelPath);
    static native void free(long ctx);
    /** prompt: optional previous text used as context (may be null). */
    static native String transcribe(long ctx, float[] pcm16k, int threads, String lang, String prompt);
    static native String systemInfo();
    static native void loadBackends(String nativeLibDir);
    /** {wallMs, melMs, encodeMs, decodeMs, sampleMs, textTokens, encoderRuns} for the last transcribe() */
    static native double[] timings(long ctx);
    static native String backendInfo();
}
