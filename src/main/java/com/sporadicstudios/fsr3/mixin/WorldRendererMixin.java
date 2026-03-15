package com.sporadicstudios.fsr3.mixin;

import net.minecraft.client.render.WorldRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import com.sporadicstudios.fsr3.client.FSR3Client;

/**
 * Mixin for WorldRenderer to hook into world rendering and apply FSR3.
 */
@Mixin(WorldRenderer.class)
public class WorldRendererMixin {

    @Inject(method = "setupTerrain", at = @At("HEAD"))
    private void onTerrainSetupStart(CallbackInfo ci) {
        // Hook for pre-terrain setup operations if needed
    }

    @Inject(method = "setupTerrain", at = @At("TAIL"))
    private void onTerrainSetupEnd(CallbackInfo ci) {
        // Hook for post-terrain setup operations if needed
    }
}
