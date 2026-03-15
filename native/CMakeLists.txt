cmake_minimum_required(VERSION 3.20)
project(vulkan_fsr3_bridge LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(vulkan_fsr3_bridge SHARED src/fsr3_bridge.cpp)

target_compile_definitions(vulkan_fsr3_bridge PRIVATE VK_USE_PLATFORM_XLIB_KHR)

target_include_directories(vulkan_fsr3_bridge PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Wire AMD FidelityFX SDK / FSR3 source here when shipping production builds.
# This sample keeps a stable JNI ABI while allowing per-platform implementations.
