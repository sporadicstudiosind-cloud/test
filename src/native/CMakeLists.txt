cmake_minimum_required(VERSION 3.16)
project(fsr3_jni)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Vulkan
find_package(Vulkan REQUIRED)

# Include directories
include_directories(${JAVA_INCLUDE_PATH})
include_directories(${JAVA_INCLUDE_PATH2})
include_directories(${Vulkan_INCLUDE_DIRS})

# Source files
set(SOURCES
    fsr3_jni.cpp
)

# Create shared library
add_library(fsr3_jni SHARED ${SOURCES})

# Link libraries
target_link_libraries(fsr3_jni PUBLIC ${Vulkan_LIBRARIES})

# Set output directory
set_target_properties(fsr3_jni PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)

# Platform-specific settings
if(WIN32)
    set_target_properties(fsr3_jni PROPERTIES
        SUFFIX ".dll"
        PREFIX ""
    )
elseif(APPLE)
    set_target_properties(fsr3_jni PROPERTIES
        SUFFIX ".dylib"
        PREFIX "lib"
    )
else()
    set_target_properties(fsr3_jni PROPERTIES
        SUFFIX ".so"
        PREFIX "lib"
    )
endif()
