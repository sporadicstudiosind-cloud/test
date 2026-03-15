#pragma once

#include <jni.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * FSR3 JNI Interface Header
 * Provides native Vulkan bindings for FSR3 upscaling and frame generation
 */

/* FSR3 Context Structure */
typedef struct {
    VkInstance instance;
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkCommandPool commandPool;
    VkQueue graphicsQueue;
    uint32_t graphicsQueueFamily;
} FSR3Context;

/* Function declarations */

/**
 * Initialize Vulkan interop
 * Returns 0 on success, non-zero on failure
 */
JNIEXPORT jint JNICALL
Java_com_sporadicstudios_fsr3_fsr3_FSR3VulkanInterop_nativeInitializeVulkan
  (JNIEnv *, jobject);

/**
 * Shutdown Vulkan interop
 * Returns 0 on success, non-zero on failure
 */
JNIEXPORT jint JNICALL
Java_com_sporadicstudios_fsr3_fsr3_FSR3VulkanInterop_nativeShutdownVulkan
  (JNIEnv *, jobject);

/**
 * Create texture handle from Minecraft texture
 * Returns texture handle on success, 0 on failure
 */
JNIEXPORT jlong JNICALL
Java_com_sporadicstudios_fsr3_fsr3_FSR3VulkanInterop_nativeCreateTexture
  (JNIEnv *, jobject, jlong);

/**
 * Get available GPU memory
 * Returns available memory in bytes
 */
JNIEXPORT jlong JNICALL
Java_com_sporadicstudios_fsr3_fsr3_FSR3VulkanInterop_nativeGetAvailableMemory
  (JNIEnv *, jobject);

/**
 * Perform FSR3 upscaling via Vulkan
 * Returns 0 on success, non-zero on failure
 */
JNIEXPORT jint JNICALL
Java_com_sporadicstudios_fsr3_fsr3_FSR3Upscaler_nativeUpscaleVulkan
  (JNIEnv *, jobject, jlong, jlong, jint, jint, jint, jint, jfloat, jfloat, jfloat);

/**
 * Fallback upscaling without advanced FSR3
 * Returns 0 on success, non-zero on failure
 */
JNIEXPORT jint JNICALL
Java_com_sporadicstudios_fsr3_fsr3_FSR3Upscaler_nativeUpscaleFallback
  (JNIEnv *, jobject, jlong, jlong, jint, jint, jint, jint);

/**
 * Perform FSR3 frame generation via Vulkan
 * Returns 0 on success, non-zero on failure
 */
JNIEXPORT jint JNICALL
Java_com_sporadicstudios_fsr3_fsr3_FSR3FrameGenerator_nativeGenerateFrameVulkan
  (JNIEnv *, jobject, jlong, jint, jint, jint, jint);

/**
 * Fallback frame generation
 * Returns 0 on success, non-zero on failure
 */
JNIEXPORT jint JNICALL
Java_com_sporadicstudios_fsr3_fsr3_FSR3FrameGenerator_nativeGenerateFrameFallback
  (JNIEnv *, jobject, jlong, jint, jint, jint);

#ifdef __cplusplus
}
#endif
