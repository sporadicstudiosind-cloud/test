package com.example.vulkanfsr3;

public final class GameRendererMixinEntrypoint {
    private GameRendererMixinEntrypoint() {
    }

    public static void onRenderFrame(float tickDelta, int width, int height) {
        Fsr3Runtime.get().onFrame(tickDelta, width, height);
    }
}
