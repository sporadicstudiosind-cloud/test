package com.sporadicstudios.fsr3.client;

import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.screen.v1.ScreenEvents;
import net.minecraft.client.MinecraftClient;
import net.minecraft.client.toast.SystemToast;
import net.minecraft.text.Text;
import com.sporadicstudios.fsr3.FSR3Mod;
import com.sporadicstudios.fsr3.config.FSR3Config;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Handles FSR3 event callbacks and keybinding events
 */
public class FSR3EventHandler {
    private static final Logger LOGGER = LoggerFactory.getLogger(FSR3Mod.MOD_ID);
    private static boolean showDebugOverlay = false;

    public static void register() {
        LOGGER.info("Registering FSR3 event handlers...");

        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            if (client.player == null) return;
            handleKeybindings(client);
        });

        LOGGER.info("FSR3 event handlers registered");
    }

    private static void handleKeybindings(MinecraftClient client) {
        if (FSR3Client.config == null || FSR3Client.fsr3Engine == null) return;

        // Toggle FSR3
        if (FSR3KeyBindings.toggleFSR3.wasPressed()) {
            boolean upscalingEnabled = FSR3Client.config.upscalingEnabled;
            boolean frameGenEnabled = FSR3Client.config.frameGenerationEnabled;
            boolean bothEnabled = upscalingEnabled && frameGenEnabled;

            FSR3Client.config.upscalingEnabled = !upscalingEnabled;
            FSR3Client.config.frameGenerationEnabled = !frameGenEnabled;
            FSR3Client.config.save();

            String message = (FSR3Client.config.upscalingEnabled || FSR3Client.config.frameGenerationEnabled)
                ? "FSR3 Enabled"
                : "FSR3 Disabled";
            sendMessage(client, message);
        }

        // Toggle Frame Generation
        if (FSR3KeyBindings.toggleFrameGeneration.wasPressed()) {
            FSR3Client.config.frameGenerationEnabled = !FSR3Client.config.frameGenerationEnabled;
            FSR3Client.config.save();
            sendMessage(client, FSR3Client.config.frameGenerationEnabled ? "Frame Generation Enabled" : "Frame Generation Disabled");
        }

        // Toggle Upscaling
        if (FSR3KeyBindings.toggleUpscaling.wasPressed()) {
            FSR3Client.config.upscalingEnabled = !FSR3Client.config.upscalingEnabled;
            FSR3Client.config.save();
            sendMessage(client, FSR3Client.config.upscalingEnabled ? "Upscaling Enabled" : "Upscaling Disabled");
        }

        // Cycle Upscaling Mode
        if (FSR3KeyBindings.cycleUpscalingMode.wasPressed()) {
            FSR3Config.FSR3UpscalingMode[] modes = FSR3Config.FSR3UpscalingMode.values();
            int currentIndex = FSR3Client.config.upscalingMode.ordinal();
            int nextIndex = (currentIndex + 1) % modes.length;
            FSR3Client.config.upscalingMode = modes[nextIndex];
            FSR3Client.config.save();
            sendMessage(client, "Upscaling Mode: " + FSR3Client.config.upscalingMode.displayName);
        }

        // Show Debug Info
        if (FSR3KeyBindings.showDebugInfo.wasPressed()) {
            showDebugOverlay = !showDebugOverlay;
            FSR3Client.config.showDebugInfo = showDebugOverlay;
            FSR3Client.config.save();
            sendMessage(client, showDebugOverlay ? "Debug Info: ON" : "Debug Info: OFF");
        }

        // Increase Sharpness
        if (FSR3KeyBindings.increaseSharpness.wasPressed()) {
            FSR3Client.config.upscalingSharpness = Math.min(3.0f, FSR3Client.config.upscalingSharpness + 0.25f);
            FSR3Client.config.save();
            sendMessage(client, String.format("Sharpness: %.2f", FSR3Client.config.upscalingSharpness));
        }

        // Decrease Sharpness
        if (FSR3KeyBindings.decreaseSharpness.wasPressed()) {
            FSR3Client.config.upscalingSharpness = Math.max(0.0f, FSR3Client.config.upscalingSharpness - 0.25f);
            FSR3Client.config.save();
            sendMessage(client, String.format("Sharpness: %.2f", FSR3Client.config.upscalingSharpness));
        }
    }

    private static void sendMessage(MinecraftClient client, String message) {
        if (client.getToastManager() != null) {
            client.getToastManager().add(
                SystemToast.create(client, SystemToast.Type.NARRATOR_TOGGLE, Text.literal(message), Text.literal("FSR3 Mod"))
            );
        }
    }

    public static boolean isDebugOverlayShown() {
        return showDebugOverlay;
    }
}
