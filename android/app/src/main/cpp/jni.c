#include <jni.h>
#include <android/log.h>
#include <stdlib.h>
#include <string.h>
#include "whisper.h"
#include "ggml-backend.h"
#include <stdio.h>
#include <time.h>
void asrllm_get_timing_totals(struct whisper_context * ctx, double * out);
static double g_wall_ms = 0, g_first_ms = -1, g_post_ms = 0, g_w0 = 0; static int g_tokens = 0;
static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000.0 + t.tv_nsec / 1e6; }

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

/* Called by whisper before each token is sampled: the first call = first token ready. */
static void first_token_cb(struct whisper_context *c, struct whisper_state *st, const whisper_token_data *td,
                           int n, float *logits, void *ud) {
    (void) c; (void) st; (void) td; (void) n; (void) logits; (void) ud;
    if (g_first_ms < 0) g_first_ms = now_ms() - g_w0;
}

JNIEXPORT jstring JNICALL
Java_com_asrllm_Whisper_transcribe(JNIEnv *env, jclass cls, jlong ctxp, jfloatArray audio, jint threads, jstring lang, jstring prompt) {
    struct whisper_context *ctx = (struct whisper_context *) ctxp;
    const char *l = (*env)->GetStringUTFChars(env, lang, NULL);
    const char *pr = prompt ? (*env)->GetStringUTFChars(env, prompt, NULL) : NULL;
    jfloat *pcm = (*env)->GetFloatArrayElements(env, audio, NULL);
    jsize n = (*env)->GetArrayLength(env, audio);

    struct whisper_full_params wp = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wp.print_realtime = false; wp.print_progress = false;
    wp.print_timestamps = false; wp.print_special = false;
    wp.translate = false; wp.language = l;
    wp.n_threads = threads; wp.no_context = true; wp.single_segment = false;
    wp.no_timestamps = true;
    if (pr && pr[0]) wp.initial_prompt = pr;   // previous text as context (live mode)
    wp.logits_filter_callback = first_token_cb;
    wp.greedy.best_of = 1;            // plain greedy decoding (fastest)
    wp.temperature_inc = 0.0f;        // no slow temperature-fallback re-decoding

    whisper_reset_timings(ctx);
    double w0 = now_ms(); g_w0 = w0; g_first_ms = -1;
    int rc = whisper_full(ctx, wp, pcm, n);
    g_wall_ms = now_ms() - w0;
    double p0 = now_ms();
    g_tokens = 0;
    if (rc == 0) {
        int nseg = whisper_full_n_segments(ctx);
        for (int i = 0; i < nseg; i++) {
            int nt = whisper_full_n_tokens(ctx, i);
            for (int j = 0; j < nt; j++)
                if (whisper_full_get_token_id(ctx, i, j) < whisper_token_eot(ctx)) g_tokens++;  // text tokens only
        }
    }
    (*env)->ReleaseFloatArrayElements(env, audio, pcm, JNI_ABORT);
    (*env)->ReleaseStringUTFChars(env, lang, l);
    if (pr) (*env)->ReleaseStringUTFChars(env, prompt, pr);
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
    jstring res = (jstring) (*env)->NewObject(env, sc, ctor, bytes, enc);
    g_post_ms = now_ms() - p0;   // token counting + text assembly + Java string
    return res;
}

JNIEXPORT jstring JNICALL
Java_com_asrllm_Whisper_systemInfo(JNIEnv *env, jclass cls) {
    return (*env)->NewStringUTF(env, whisper_print_system_info());
}

JNIEXPORT void JNICALL
Java_com_asrllm_Whisper_loadBackends(JNIEnv *env, jclass cls, jstring dir) {
    const char *d = (*env)->GetStringUTFChars(env, dir, NULL);
    ggml_backend_load_all_from_path(d);
    (*env)->ReleaseStringUTFChars(env, dir, d);
}

/** Totals for the last whisper_full call (ms unless noted):
 * 0 wall(whisper_full) 1 mel 2 encode 3 decode total(prompt+batch+single) 4 sampling 5 text tokens 6 encoder runs
 * 7 single-token decode 8 batch decode 9 prompt decode 10 n single decode calls 11 n batch calls
 * 12 first token ready (ms from start, -1 if none) 13 text assembly (post) */
JNIEXPORT jdoubleArray JNICALL
Java_com_asrllm_Whisper_timings(JNIEnv *env, jclass cls, jlong ctxp) {
    double t[11]; asrllm_get_timing_totals((struct whisper_context *) ctxp, t);
    double v[14] = { g_wall_ms, t[0], t[2], t[3] + t[4] + t[5], t[1], (double) g_tokens, t[7],
                     t[3], t[4], t[5], t[8], t[9], g_first_ms, g_post_ms };
    jdoubleArray a = (*env)->NewDoubleArray(env, 14);
    (*env)->SetDoubleArrayRegion(env, a, 0, 14, v);
    return a;
}

/** Names of loaded ggml backends/devices (shows which CPU variant was picked). */
JNIEXPORT jstring JNICALL
Java_com_asrllm_Whisper_backendInfo(JNIEnv *env, jclass cls) {
    char buf[1024]; size_t o = 0; buf[0] = 0;
    for (size_t i = 0; i < ggml_backend_reg_count() && o < sizeof(buf) - 1; i++) {
        ggml_backend_reg_t r = ggml_backend_reg_get(i);
        o += snprintf(buf + o, sizeof(buf) - o, "%s%s", i ? ", " : "", ggml_backend_reg_name(r));
    }
    for (size_t i = 0; i < ggml_backend_dev_count() && o < sizeof(buf) - 1; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        o += snprintf(buf + o, sizeof(buf) - o, " | dev: %s", ggml_backend_dev_description(d));
    }
    return (*env)->NewStringUTF(env, buf);
}
