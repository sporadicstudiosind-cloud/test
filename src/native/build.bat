@echo off
REM FSR3 JNI Native Build Script for Windows

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set BUILD_DIR=%SCRIPT_DIR%build
set OUTPUT_DIR=%SCRIPT_DIR%output

echo [FSR3] Building native code...

REM Create build directories
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Enter build directory
cd /d "%BUILD_DIR%"

REM Run CMake
echo [FSR3] Running CMake...
cmake .. ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DJAVA_INCLUDE_PATH="%JAVA_HOME%\include" ^
    -DJAVA_INCLUDE_PATH2="%JAVA_HOME%\include\win32" ^
    -G "Visual Studio 17 2022" ^
    -A x64

if errorlevel 1 (
    echo [FSR3] CMake failed!
    exit /b 1
)

REM Build
echo [FSR3] Building...
cmake --build . --config Release

if errorlevel 1 (
    echo [FSR3] Build failed!
    exit /b 1
)

REM Copy output
echo [FSR3] Copying outputs...
copy Release\*.dll "%OUTPUT_DIR%" >nul 2>&1

echo [FSR3] Build complete! Output in: %OUTPUT_DIR%
dir "%OUTPUT_DIR%"

endlocal
