#include <jni.h>
#include <atomic>

namespace {
std::atomic<bool> g_enabled{false};
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_vulkanfsr3_Fsr3Runtime_nativeSetEnabled(JNIEnv*, jclass, jboolean enabled) {
    g_enabled.store(enabled == JNI_TRUE, std::memory_order_relaxed);
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_vulkanfsr3_Fsr3Runtime_nativeOnFrame(JNIEnv*, jclass, jfloat /*deltaTime*/, jint /*width*/, jint /*height*/) {
    if (!g_enabled.load(std::memory_order_relaxed)) {
        return;
    }

    // Production implementation notes:
    // 1) Acquire Vulkan image handles from VulkanMod render graph.
    // 2) Dispatch FSR3 upscaling pass using FidelityFX Super Resolution 3 API.
    // 3) Run frame generation pass and present interpolated frame.
    // 4) Handle swapchain resize/device lost.
}
