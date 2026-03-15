package com.sporadicstudios.fsr3.fsr3;

import com.sporadicstudios.fsr3.FSR3Mod;
import com.sporadicstudios.fsr3.config.FSR3Config;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main FSR3 Engine that manages FSR3 upscaling and frame generation.
 * This acts as a bridge between Minecraft's rendering pipeline and VulkanMod's capabilities.
 */
public class FSR3Engine {
    private static final Logger LOGGER = LoggerFactory.getLogger(FSR3Mod.MOD_ID);

    private FSR3Upscaler upscaler;
    private FSR3FrameGenerator frameGenerator;
    private FSR3VulkanInterop vulkanInterop;
    private boolean initialized = false;

    public FSR3Engine() {
        this.upscaler = new FSR3Upscaler();
        this.frameGenerator = new FSR3FrameGenerator();
        this.vulkanInterop = new FSR3VulkanInterop();
    }

    public void initialize() throws Exception {
        LOGGER.info("Initializing FSR3 Engine...");

        // Initialize Vulkan interop
        try {
            vulkanInterop.initialize();
            LOGGER.info("Vulkan interop initialized");
        } catch (Exception e) {
            LOGGER.warn("Vulkan interop initialization failed, FSR3 may not work optimally", e);
        }

        // Initialize upscaler
        upscaler.initialize(vulkanInterop);
        LOGGER.info("FSR3 Upscaler initialized");

        // Initialize frame generator
        frameGenerator.initialize(vulkanInterop);
        LOGGER.info("FSR3 Frame Generator initialized");

        initialized = true;
        LOGGER.info("FSR3 Engine fully initialized and ready");
    }

    public void shutdown() {
        if (!initialized) return;

        try {
            if (frameGenerator != null) {
                frameGenerator.shutdown();
            }
            if (upscaler != null) {
                upscaler.shutdown();
            }
            if (vulkanInterop != null) {
                vulkanInterop.shutdown();
            }
            initialized = false;
            LOGGER.info("FSR3 Engine shut down");
        } catch (Exception e) {
            LOGGER.error("Error during FSR3 Engine shutdown", e);
        }
    }

    public boolean isInitialized() {
        return initialized;
    }

    public FSR3Upscaler getUpscaler() {
        return upscaler;
    }

    public FSR3FrameGenerator getFrameGenerator() {
        return frameGenerator;
    }

    public FSR3VulkanInterop getVulkanInterop() {
        return vulkanInterop;
    }

    /**
     * Process a frame through FSR3 pipeline
     */
    public void processFrame(long inputTextureHandle, long outputTextureHandle, int inputWidth, int inputHeight,
                            int outputWidth, int outputHeight, FSR3Config config) {
        if (!initialized) return;

        try {
            // Apply upscaling if enabled
            if (config.upscalingEnabled) {
                upscaler.upscale(
                    inputTextureHandle,
                    outputTextureHandle,
                    inputWidth,
                    inputHeight,
                    outputWidth,
                    outputHeight,
                    config.upscalingMode.renderScale,
                    config.upscalingSharpness
                );
            }

            // Apply frame generation if enabled
            if (config.frameGenerationEnabled) {
                frameGenerator.generateFrame(
                    outputTextureHandle,
                    outputWidth,
                    outputHeight,
                    config.frameGenerationMode,
                    config.useMotionEstimation
                );
            }
        } catch (Exception e) {
            LOGGER.error("Error processing frame through FSR3", e);
        }
    }

    public void onFrameStart() {
        if (initialized && frameGenerator != null) {
            frameGenerator.onFrameStart();
        }
    }

    public void onFrameEnd() {
        if (initialized && frameGenerator != null) {
            frameGenerator.onFrameEnd();
        }
    }
}
