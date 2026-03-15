package com.example.vulkanfsr3;

import java.util.concurrent.atomic.AtomicBoolean;

public final class Fsr3Runtime {
    private static final Fsr3Runtime INSTANCE = new Fsr3Runtime();
    private static final String LIB_NAME = "vulkan_fsr3_bridge";

    private final AtomicBoolean enabled = new AtomicBoolean(false);
    private volatile boolean nativeLoaded = false;

    private Fsr3Runtime() {
    }

    public static Fsr3Runtime get() {
        return INSTANCE;
    }

    public static void loadNativeIfPresent() {
        try {
            System.loadLibrary(LIB_NAME);
            get().nativeLoaded = true;
            VulkanFsr3BridgeMod.LOGGER.info("Loaded native FSR3 bridge library: {}", LIB_NAME);
        } catch (UnsatisfiedLinkError ex) {
            VulkanFsr3BridgeMod.LOGGER.warn("Native FSR3 bridge library not found; running fallback mode.");
        }
    }

    public void setEnabled(boolean enabledState) {
        enabled.set(enabledState);
        if (nativeLoaded) {
            nativeSetEnabled(enabledState);
        }
    }

    public String statusLine() {
        return "FSR3 enabled=" + enabled.get() + ", native=" + nativeLoaded;
    }

    public void onFrame(float deltaTime, int width, int height) {
        if (!enabled.get() || !nativeLoaded) {
            return;
        }
        nativeOnFrame(deltaTime, width, height);
    }

    private static native void nativeSetEnabled(boolean enabled);
    private static native void nativeOnFrame(float deltaTime, int width, int height);
}
