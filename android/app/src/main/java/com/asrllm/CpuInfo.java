package com.asrllm;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Map;
import java.util.TreeMap;

/** Reads CPU topology/features from /sys and /proc (works on all Android versions). */
final class CpuInfo {
    private static long maxFreq(int cpu) {
        try (BufferedReader r = new BufferedReader(new FileReader("/sys/devices/system/cpu/cpu" + cpu + "/cpufreq/cpuinfo_max_freq"))) {
            return Long.parseLong(r.readLine().trim());
        } catch (Throwable t) { return -1; }
    }

    /** max-frequency (MHz) -> core count */
    static Map<Long, Integer> clusterMap() {
        Map<Long, Integer> m = new TreeMap<>();
        int n = Runtime.getRuntime().availableProcessors();
        for (int i = 0; i < n; i++) { long f = maxFreq(i); if (f > 0) { Long k = f / 1000; Integer c = m.get(k); m.put(k, c == null ? 1 : c + 1); } }
        return m;
    }

    static String clusters() {
        Map<Long, Integer> m = clusterMap();
        if (m.isEmpty()) return "clusters: unknown";
        StringBuilder sb = new StringBuilder("clusters:");
        for (Map.Entry<Long, Integer> e : m.entrySet()) sb.append(' ').append(e.getValue()).append("x").append(e.getKey()).append("MHz");
        return sb.toString();
    }

    /** Number of non-"little" cores (all cores except the slowest cluster), clamped to 1..8. */
    static int defaultThreads() {
        int n = Runtime.getRuntime().availableProcessors();
        Map<Long, Integer> m = clusterMap();
        int t;
        if (m.size() >= 2) {
            int little = m.values().iterator().next();
            t = n - little;
            if (t < 2) t = Math.min(4, n);
        } else {
            t = Math.min(4, n);
        }
        return Math.max(1, Math.min(8, t));
    }

    static String features() {
        try (BufferedReader r = new BufferedReader(new FileReader(new File("/proc/cpuinfo")))) {
            String line;
            while ((line = r.readLine()) != null) {
                String l = line.toLowerCase(java.util.Locale.ROOT);
                if (l.startsWith("features") || l.startsWith("flags")) {
                    String f = line.substring(line.indexOf(':') + 1).trim();
                    StringBuilder sb = new StringBuilder();
                    for (String k : new String[]{"asimd", "asimdhp", "fphp", "asimddp", "i8mm", "sve", "sve2", "sme", "bf16", "avx2", "avx512f"})
                        if ((" " + f + " ").contains(" " + k + " ")) sb.append(k).append(' ');
                    return sb.length() > 0 ? sb.toString().trim() : f;
                }
            }
        } catch (Throwable ignored) {}
        return "unknown";
    }
}
