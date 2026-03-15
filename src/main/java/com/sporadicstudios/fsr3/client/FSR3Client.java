package com.sporadicstudios.fsr3.client;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.api.EnvType;
import net.fabricmc.api.Environment;
import com.sporadicstudios.fsr3.FSR3Mod;
import com.sporadicstudios.fsr3.config.FSR3Config;
import com.sporadicstudios.fsr3.fsr3.FSR3Engine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Environment(EnvType.CLIENT)
public class FSR3Client implements ClientModInitializer {
    private static final Logger LOGGER = LoggerFactory.getLogger(FSR3Mod.MOD_ID);
    public static FSR3Engine fsr3Engine;
    public static FSR3Config config;

    @Override
    public void onInitializeClient() {
        LOGGER.info("FSR3 Client initializing...");

        // Load configuration
        config = new FSR3Config();
        config.load();

        // Initialize FSR3 Engine
        try {
            fsr3Engine = new FSR3Engine();
            fsr3Engine.initialize();
            LOGGER.info("FSR3 Engine initialized successfully");
        } catch (Exception e) {
            LOGGER.error("Failed to initialize FSR3 Engine", e);
        }

        // Register key bindings
        FSR3KeyBindings.register();

        // Register event handlers
        FSR3EventHandler.register();

        LOGGER.info("FSR3 Client initialized");
    }
}
