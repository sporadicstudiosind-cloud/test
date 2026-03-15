package com.example.vulkanfsr3.mixin;

import com.example.vulkanfsr3.GameRendererMixinEntrypoint;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.render.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(GameRenderer.class)
public abstract class GameRendererMixin {
    @Inject(method = "render", at = @At("TAIL"))
    private void vulkanFsr3Bridge$afterRender(float tickDelta, long startTime, boolean tick, CallbackInfo ci) {
        MinecraftClient client = MinecraftClient.getInstance();
        if (client.getWindow() == null) {
            return;
        }
        GameRendererMixinEntrypoint.onRenderFrame(tickDelta, client.getWindow().getFramebufferWidth(), client.getWindow().getFramebufferHeight());
    }
}
