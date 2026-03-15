# FSR3 Minecraft Mod

A high-performance Minecraft mod implementing AMD FSR3 Frame Generation and Upscaling for enhanced visuals and performance.

## Features

- **AMD FSR3 Upscaling**: Render at lower resolutions with high-quality upscaling to improve performance while maintaining visual fidelity
- **AMD FSR3 Frame Generation**: Generate interpolated frames to increase perceived framerate
- **VulkanMod Integration**: Leverages VulkanMod for GPU-accelerated rendering
- **Configurable Settings**: Fine-tune upscaling modes, sharpness, and frame generation parameters
- **Real-time Performance Monitoring**: View FPS and performance metrics in-game
- **Keybindings**: Easy in-game control of FSR3 features

## Requirements

- **Minecraft Version**: 1.21
- **Loader**: Fabric Loader 0.15+
- **Fabric API**: 0.97.0+
- **VulkanMod**: 0.3.95 (Fabric)
- **Java**: 17+

## Installation

1. Download the latest JAR from [releases](https://github.com/sporadicstudiosind/fsr3-minecraft-mod/releases)
2. Place the JAR file in your `mods/` directory
3. Install [VulkanMod](https://www.curseforge.com/minecraft/mods/vulkanmod) if not already installed
4. Launch Minecraft with Fabric

## Building from Source

### Prerequisites

- JDK 17 or higher
- Gradle (included via wrapper)

### Compile

```bash
./gradlew build
```

The compiled JAR will be in `build/libs/fsr3-minecraft-mod-1.0.0.jar`

## Configuration

Configuration file: `config/fsr3-config.json`

### Available Settings

```json
{
  "upscalingEnabled": true,
  "upscalingMode": "BALANCED",
  "upscalingSharpness": 2.0,
  "frameGenerationEnabled": true,
  "useMotionEstimation": true,
  "frameGenerationMode": 1,
  "enableVsync": true,
  "targetFramerate": 60,
  "showDebugInfo": false
}
```

**Upscaling Modes:**
- `PERFORMANCE`: 50% render scale (2x upscaling)
- `BALANCED`: 66.7% render scale
- `QUALITY`: 77.8% render scale
- `ULTRA_QUALITY`: 88.9% render scale

**Frame Generation Modes:**
- `0`: Conservative (lower overhead, less interpolation)
- `1`: Balanced (recommended)
- `2`: Aggressive (maximum interpolation, higher overhead)

## Keybindings

- **F9**: Toggle FSR3
- **F10**: Toggle Frame Generation
- **F11**: Toggle Upscaling
- **F12**: Cycle Upscaling Mode
- **P**: Show Debug Information
- **[**: Decrease Sharpness
- **]**: Increase Sharpness

## Performance Impact

- **Upscaling**: Significant FPS improvement (20-40% depending on mode)
- **Frame Generation**: Perceived FPS boost without full frame renders
- **Combined**: Best performance with minimal visual quality loss

## Troubleshooting

### FSR3 not working
- Ensure VulkanMod is installed and updated
- Check that your GPU supports Vulkan
- Try disabling and re-enabling FSR3

### Performance issues
- Try using Performance upscaling mode
- Disable frame generation if stuttering occurs
- Check your GPU drivers are up-to-date

### Crashes on startup
- Verify Java 17+ is installed
- Check VulkanMod compatibility with your Minecraft version
- Review latest log in `logs/latest.log`

## Known Issues

- Updating VulkanMod may break compatibility (requires mod recompilation)
- Some shaders may not be fully compatible with FSR3
- Motion estimation can have visual artifacts in fast motion scenes

## Development

This mod uses JNI for native Vulkan integration. Native code is located in `src/native/`.

### Building Native Code

```bash
cd src/native
./build.sh
```

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/sporadicstudiosind/fsr3-minecraft-mod/issues
- CurseForge: https://www.curseforge.com/minecraft/mods/fsr3-minecraft-mod

## Credits

- AMD FSR3 Technology
- VulkanMod Developers
- Fabric Team

---

**Disclaimer**: This mod modifies game rendering behavior. Use at your own risk and ensure compatibility with other mods before reporting issues.
