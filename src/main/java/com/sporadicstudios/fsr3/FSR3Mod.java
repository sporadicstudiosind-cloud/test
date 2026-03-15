package com.sporadicstudios.fsr3;

import net.fabricmc.api.ModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FSR3Mod implements ModInitializer {
    public static final String MOD_ID = "fsr3mod";
    public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

    @Override
    public void onInitialize() {
        LOGGER.info("FSR3 Mod initializing...");
        LOGGER.info("AMD FSR3 Frame Generation and Upscaling enabled");
    }
}
