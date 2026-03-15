#!/bin/bash

# FSR3 JNI Native Build Script

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"
OUTPUT_DIR="$SCRIPT_DIR/output"

echo "[FSR3] Building native code..."

# Create build directories
mkdir -p "$BUILD_DIR"
mkdir -p "$OUTPUT_DIR"

# Enter build directory
cd "$BUILD_DIR"

# Run CMake
echo "[FSR3] Running CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DJAVA_INCLUDE_PATH="$JAVA_HOME/include" \
    -DJAVA_INCLUDE_PATH2="$JAVA_HOME/include/linux"

# Build
echo "[FSR3] Building..."
make -j$(nproc)

# Copy output
echo "[FSR3] Copying outputs..."
cp lib/* "$OUTPUT_DIR/" 2>/dev/null || true

echo "[FSR3] Build complete! Output in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
