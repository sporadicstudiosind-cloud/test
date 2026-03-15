# Vulkan FSR3 Bridge (Fabric)

Experimental Fabric client mod that pins a specific VulkanMod release and exposes a JNI bridge for AMD FSR3 upscaling + frame generation.

## Scope

This repository ships:

- Fabric mod scaffold with fixed `vulkanmod` dependency.
- Runtime toggle commands (`/fsr3 enable`, `/fsr3 disable`, `/fsr3 status`).
- Render-loop hook (`GameRenderer#render`) to call into a native library.
- Native C++ JNI ABI surface where FidelityFX FSR3 integration can be implemented.

## Build

```bash
./gradlew build
```

## Native bridge build

```bash
cd native
cmake -S . -B build
cmake --build build --config Release
```

Place built `libvulkan_fsr3_bridge`/`vulkan_fsr3_bridge.dll` on `java.library.path` before launching Minecraft.

## Stability / compatibility

- `vulkanmod` is hard-pinned in both Gradle dependency and `fabric.mod.json` dependency metadata.
- Updating VulkanMod can break the bridge and is intentionally unsupported.
