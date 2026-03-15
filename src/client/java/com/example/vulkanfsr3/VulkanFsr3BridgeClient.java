package com.example.vulkanfsr3;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandManager;
import net.fabricmc.fabric.api.client.command.v2.ClientCommandRegistrationCallback;
import net.minecraft.text.Text;

public class VulkanFsr3BridgeClient implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        Fsr3Runtime.loadNativeIfPresent();

        ClientCommandRegistrationCallback.EVENT.register((dispatcher, registryAccess) -> {
            dispatcher.register(ClientCommandManager.literal("fsr3")
                .then(ClientCommandManager.literal("enable").executes(ctx -> {
                    Fsr3Runtime.get().setEnabled(true);
                    ctx.getSource().sendFeedback(Text.literal("FSR3 enabled"));
                    return 1;
                }))
                .then(ClientCommandManager.literal("disable").executes(ctx -> {
                    Fsr3Runtime.get().setEnabled(false);
                    ctx.getSource().sendFeedback(Text.literal("FSR3 disabled"));
                    return 1;
                }))
                .then(ClientCommandManager.literal("status").executes(ctx -> {
                    ctx.getSource().sendFeedback(Text.literal(Fsr3Runtime.get().statusLine()));
                    return 1;
                }))
            );
        });
    }
}
