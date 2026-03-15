package com.example.vulkanfsr3;

import net.fabricmc.api.ModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class VulkanFsr3BridgeMod implements ModInitializer {
    public static final String MOD_ID = "vulkan-fsr3-bridge";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitialize() {
        LOGGER.info("Initializing Vulkan FSR3 Bridge");
    }
}
