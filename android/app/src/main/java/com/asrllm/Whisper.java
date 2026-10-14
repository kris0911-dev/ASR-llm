package com.asrllm;

/** Thin JNI wrapper around whisper.cpp. Call from one background thread only. */
final class Whisper {
    static { System.loadLibrary("asrllm"); }
    static native long init(String modelPath);
    static native void free(long ctx);
    static native String transcribe(long ctx, float[] pcm16k, int threads, String lang);
    static native String systemInfo();
}
