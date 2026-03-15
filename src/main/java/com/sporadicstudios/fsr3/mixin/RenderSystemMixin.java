package com.sporadicstudios.fsr3.mixin;

import net.minecraft.client.render.RenderSystem;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import com.sporadicstudios.fsr3.client.FSR3Client;

/**
 * Mixin for RenderSystem to hook into low-level rendering operations.
 */
@Mixin(RenderSystem.class)
public class RenderSystemMixin {

    @Inject(method = "beginFrame", at = @At("HEAD"))
    private static void onBeginFrame(CallbackInfo ci) {
        // Hook for frame start operations
        if (FSR3Client.fsr3Engine != null && FSR3Client.fsr3Engine.isInitialized()) {
            // Can perform pre-render setup here
        }
    }

    @Inject(method = "endFrame", at = @At("HEAD"))
    private static void onEndFrame(CallbackInfo ci) {
        // Hook for frame end operations
        if (FSR3Client.fsr3Engine != null && FSR3Client.fsr3Engine.isInitialized()) {
            // Can perform post-render cleanup here
        }
    }
}
