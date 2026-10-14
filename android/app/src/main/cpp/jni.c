#include <jni.h>
#include <android/log.h>
#include <stdlib.h>
#include <string.h>
#include "whisper.h"

#define TAG "asrllm"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)

JNIEXPORT jlong JNICALL
Java_com_asrllm_Whisper_init(JNIEnv *env, jclass cls, jstring path) {
    const char *p = (*env)->GetStringUTFChars(env, path, NULL);
    struct whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;
    struct whisper_context *ctx = whisper_init_from_file_with_params(p, cp);
    (*env)->ReleaseStringUTFChars(env, path, p);
    return (jlong) ctx;
}

JNIEXPORT void JNICALL
Java_com_asrllm_Whisper_free(JNIEnv *env, jclass cls, jlong ctx) {
    if (ctx) whisper_free((struct whisper_context *) ctx);
}

JNIEXPORT jstring JNICALL
Java_com_asrllm_Whisper_transcribe(JNIEnv *env, jclass cls, jlong ctxp, jfloatArray audio, jint threads, jstring lang) {
    struct whisper_context *ctx = (struct whisper_context *) ctxp;
    const char *l = (*env)->GetStringUTFChars(env, lang, NULL);
    jfloat *pcm = (*env)->GetFloatArrayElements(env, audio, NULL);
    jsize n = (*env)->GetArrayLength(env, audio);

    struct whisper_full_params wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wp.print_realtime = false; wp.print_progress = false;
    wp.print_timestamps = false; wp.print_special = false;
    wp.translate = false; wp.language = l;
    wp.n_threads = threads; wp.no_context = true; wp.single_segment = false;

    int rc = whisper_full(ctx, wp, pcm, n);
    (*env)->ReleaseFloatArrayElements(env, audio, pcm, JNI_ABORT);
    (*env)->ReleaseStringUTFChars(env, lang, l);
    if (rc != 0) return (*env)->NewStringUTF(env, "[error: whisper_full failed]");

    size_t cap = 4096, len = 0; char *buf = malloc(cap); buf[0] = 0;
    int ns = whisper_full_n_segments(ctx);
    for (int i = 0; i < ns; i++) {
        const char *t = whisper_full_get_segment_text(ctx, i);
        size_t tl = strlen(t);
        while (len + tl + 1 > cap) { cap *= 2; buf = realloc(buf, cap); }
        memcpy(buf + len, t, tl + 1); len += tl;
    }
    // Build a Java String from UTF-8 bytes (NewStringUTF rejects 4-byte UTF-8).
    jbyteArray bytes = (*env)->NewByteArray(env, (jsize) len);
    (*env)->SetByteArrayRegion(env, bytes, 0, (jsize) len, (const jbyte *) buf);
    free(buf);
    jclass sc = (*env)->FindClass(env, "java/lang/String");
    jmethodID ctor = (*env)->GetMethodID(env, sc, "<init>", "([BLjava/lang/String;)V");
    jstring enc = (*env)->NewStringUTF(env, "UTF-8");
    return (jstring) (*env)->NewObject(env, sc, ctor, bytes, enc);
}

JNIEXPORT jstring JNICALL
Java_com_asrllm_Whisper_systemInfo(JNIEnv *env, jclass cls) {
    return (*env)->NewStringUTF(env, whisper_print_system_info());
}
