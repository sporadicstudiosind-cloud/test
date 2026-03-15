package com.sporadicstudios.fsr3.client;

import net.fabricmc.fabric.api.client.keybinding.v1.KeyBindingHelper;
import net.minecraft.client.option.KeyBinding;
import net.minecraft.client.util.InputUtil;
import org.lwjgl.glfw.GLFW;
import com.sporadicstudios.fsr3.FSR3Mod;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Registers and manages FSR3 keybindings for in-game controls.
 */
public class FSR3KeyBindings {
    private static final Logger LOGGER = LoggerFactory.getLogger(FSR3Mod.MOD_ID);

    public static KeyBinding toggleFSR3;
    public static KeyBinding toggleFrameGeneration;
    public static KeyBinding toggleUpscaling;
    public static KeyBinding cycleUpscalingMode;
    public static KeyBinding showDebugInfo;
    public static KeyBinding increaseSharpness;
    public static KeyBinding decreaseSharpness;

    public static void register() {
        LOGGER.info("Registering FSR3 keybindings...");

        toggleFSR3 = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.fsr3mod.toggle_fsr3",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_F9,
            "category.fsr3mod"
        ));

        toggleFrameGeneration = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.fsr3mod.toggle_frame_generation",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_F10,
            "category.fsr3mod"
        ));

        toggleUpscaling = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.fsr3mod.toggle_upscaling",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_F11,
            "category.fsr3mod"
        ));

        cycleUpscalingMode = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.fsr3mod.cycle_upscaling_mode",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_F12,
            "category.fsr3mod"
        ));

        showDebugInfo = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.fsr3mod.show_debug_info",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_P,
            "category.fsr3mod"
        ));

        increaseSharpness = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.fsr3mod.increase_sharpness",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_RIGHT_BRACKET,
            "category.fsr3mod"
        ));

        decreaseSharpness = KeyBindingHelper.registerKeyBinding(new KeyBinding(
            "key.fsr3mod.decrease_sharpness",
            InputUtil.Type.KEYSYM,
            GLFW.GLFW_KEY_LEFT_BRACKET,
            "category.fsr3mod"
        ));

        LOGGER.info("FSR3 keybindings registered successfully");
    }
}
