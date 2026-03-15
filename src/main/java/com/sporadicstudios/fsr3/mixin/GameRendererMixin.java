package com.sporadicstudios.fsr3.mixin;

import net.minecraft.client.render.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import com.sporadicstudios.fsr3.client.FSR3Client;

/**
 * Mixin for GameRenderer to hook into render cycle and apply FSR3.
 */
@Mixin(GameRenderer.class)
public class GameRendererMixin {

    @Inject(method = "render", at = @At("HEAD"))
    private void onRenderStart(float tickDelta, long startTime, boolean tick, CallbackInfo ci) {
        if (FSR3Client.fsr3Engine != null && FSR3Client.fsr3Engine.isInitialized()) {
            FSR3Client.fsr3Engine.onFrameStart();
        }
    }

    @Inject(method = "render", at = @At("TAIL"))
    private void onRenderEnd(float tickDelta, long startTime, boolean tick, CallbackInfo ci) {
        if (FSR3Client.fsr3Engine != null && FSR3Client.fsr3Engine.isInitialized()) {
            FSR3Client.fsr3Engine.onFrameEnd();
        }
    }
}
