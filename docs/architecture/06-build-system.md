# 6. Build System

This document describes the CMake-based build system and critical compilation settings.

---

## 6.1 CMake Structure

### Top-Level Organization

```cmake
project(model_infer)
cmake_minimum_required(VERSION 3.10)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(OpenCV 4 REQUIRED)
find_package(Lua REQUIRED)

# Subdirectories
add_subdirectory(lua-intf-ex)      # LuaIntf library
add_subdirectory(onnxruntime)      # ONNX Runtime

# Source organization
set(TENSOR_SOURCES ...)
set(INFERENCE_SOURCES ...)

# Executables
add_executable(lua_runner ...)
add_executable(cpp_infer ...)

# Linking
target_link_libraries(lua_runner ...)
```

### Shared Source Lists

To avoid duplication between `lua_runner` and `cpp_infer`:

```cmake
set(TENSOR_SOURCES
    src/modules/tensor/cpu_memory.cpp
    src/modules/tensor/device_buffer.cpp
    src/modules/tensor/tensor_core.cpp
    src/modules/tensor/tensor_device.cpp
    src/modules/tensor/tensor_shape.cpp
    src/modules/tensor/tensor_math.cpp
    src/modules/tensor/tensor_activation.cpp
    src/modules/tensor/tensor_compare.cpp
    src/modules/tensor/tensor_reduction.cpp
    src/modules/tensor/tensor_select.cpp
    src/modules/tensor/tensor_advanced.cpp
    src/modules/tensor/tensor_legacy.cpp
)

set(INFERENCE_SOURCES
    src/inference/inference.cpp
)

# Both executables use these
add_executable(lua_runner
    src/main.cpp
    ${TENSOR_SOURCES}
    ${INFERENCE_SOURCES}
    ...
)

add_executable(cpp_infer
    src/cpp_main.cpp
    ${TENSOR_SOURCES}
    ${INFERENCE_SOURCES}
)
```

---

## 6.2 Critical Build Configuration

### 6.2.1 Lua Compiled as C++

**Most important build setting**:

```cmake
# In lua-intf-ex/lua/CMakeLists.txt or top-level CMakeLists.txt
target_compile_options(lua_static PRIVATE
    -x c++          # Treat .c files as C++
    -O3             # Optimization
    -Wall           # Warnings
    -DLUA_USE_POSIX # POSIX features
)
```

**Why this is critical**:

```
Without -x c++:
  C++ code → calls Lua function → Lua (C) → calls C++ callback
               │                               │
               └─ C++ exception thrown         │
                         │                     │
                         └─ Unwinds through Lua C frames
                                 │
                                 └─→ UNDEFINED BEHAVIOR!
                                     - Stack corruption
                                     - Destructors not called
                                     - Memory leaks

With -x c++:
  C++ code → calls Lua function → Lua (C++) → calls C++ callback
               │                                │
               └─ C++ exception thrown          │
                         │                      │
                         └─ Properly unwinds through Lua C++ frames
                                 │
                                 └─→ SAFE
                                     - Destructors called
                                     - Resources cleaned up
```

**Consequences if not set**:
- ❌ Exceptions thrown in C++ callbacks crash the program
- ❌ Memory leaks when exceptions cross boundary
- ❌ Stack corruption on exception unwind
- ❌ Unpredictable behavior

### 6.2.2 Optimization Levels

```cmake
# Debug build
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# Release build
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
```

**Recommendations**:
- Development: `Debug` for better error messages
- Testing: `Release` for performance validation
- Deployment: `Release` for production

---

## 6.3 Dependency Management

### 6.3.1 OpenCV

```cmake
find_package(OpenCV 4 REQUIRED)

target_link_libraries(lua_runner
    ${OpenCV_LIBS}
)

# Required OpenCV modules:
# - core (cv::Mat)
# - imgproc (cv::resize, cv::cvtColor)
# - imgcodecs (cv::imread, cv::imwrite)
# - videoio (cv::VideoCapture, cv::VideoWriter)
# - highgui (cv::imshow, cv::waitKey)
```

**Version requirement**: OpenCV 4.x (tested with 4.6.0+)

### 6.3.2 Lua

```cmake
find_package(Lua REQUIRED)

target_include_directories(lua_runner PRIVATE
    ${LUA_INCLUDE_DIR}
)

target_link_libraries(lua_runner
    lua_static  # Or ${LUA_LIBRARIES}
)
```

**Version requirement**: Lua 5.3 or 5.4

**Note**: The system uses a bundled Lua in `lua-intf-ex/lua/` compiled as C++.

### 6.3.3 ONNX Runtime

```cmake
add_subdirectory(onnxruntime)

target_link_libraries(lua_runner
    onnxruntime
)
```

**Typical setup**:
- Prebuilt ONNX Runtime library (libonnxruntime.so)
- Headers in `onnxruntime/include/`
- Library in `onnxruntime/lib/`

**Version**: 1.12.0+ recommended

### 6.3.4 LuaIntf

```cmake
add_subdirectory(lua-intf-ex)

target_link_libraries(lua_runner
    lua-intf-ex
)

target_include_directories(lua_runner PRIVATE
    lua-intf-ex/
)
```

**Source**: Custom fork with extensions in `lua-intf-ex/` directory.

---

## 6.4 Build Commands

### Standard Build

```bash
# Create build directory
mkdir build
cd build

# Configure
cmake ..

# Build
make -j$(nproc)

# Output:
#   build/lua_runner
#   build/cpp_infer
```

### Rebuild After Changes

```bash
cd build

# Rebuild changed files only
make -j$(nproc)

# Full rebuild (if needed)
make clean
make -j$(nproc)
```

### Build Targets

```bash
# Build specific target
make lua_runner
make cpp_infer

# Build tests (if defined)
make tests
```

---

## 6.5 Build Output

### Executables

```
build/
  ├── lua_runner          # Main Lua-based runner
  ├── cpp_infer           # Pure C++ benchmark
  ├── liblua_static.a     # Lua library (compiled as C++)
  ├── liblua-intf-ex.a    # LuaIntf library
  └── libonnxruntime.so   # ONNX Runtime (shared library)
```

### Running Executables

```bash
# Lua runner (from project root)
./build/lua_runner scripts/yolo11_tensor_detector.lua \
                   models/yolo11n.onnx \
                   images/zidane.jpg

# C++ benchmark
./build/cpp_infer models/yolo11n.onnx images/zidane.jpg
```

---

## 6.6 Platform-Specific Considerations

### Embedded Linux (RISC-V/ARM)

```cmake
# Cross-compilation toolchain
set(CMAKE_C_COMPILER riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)

# Target architecture
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Optimization for size (embedded)
set(CMAKE_CXX_FLAGS_RELEASE "-Os -DNDEBUG")
```

### x86_64 Linux

```cmake
# Native build
# (defaults work)

# Optional: Enable AVX2 for performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
```

---

## 6.7 Common Build Issues

### Issue: Lua compiled as C (not C++)

**Symptom**: Crashes when C++ exceptions are thrown

**Fix**: Add `-x c++` to lua compilation flags

```cmake
target_compile_options(lua_static PRIVATE -x c++)
```

### Issue: ONNX Runtime not found

**Symptom**: `onnxruntime.so: cannot open shared object file`

**Fix**: Add ONNX Runtime lib to library path

```bash
export LD_LIBRARY_PATH=/path/to/onnxruntime/lib:$LD_LIBRARY_PATH
```

Or copy library to standard location:

```bash
sudo cp onnxruntime/lib/libonnxruntime.so* /usr/local/lib/
sudo ldconfig
```

### Issue: OpenCV version mismatch

**Symptom**: Undefined references to OpenCV functions

**Fix**: Ensure OpenCV 4.x is installed

```bash
# Check version
pkg-config --modversion opencv4

# If wrong version, rebuild OpenCV or adjust CMake
find_package(OpenCV 4.6 EXACT REQUIRED)
```

---

## 6.8 Build Performance

### Parallel Compilation

```bash
# Use all CPU cores
make -j$(nproc)

# Limit to N cores (e.g., 8)
make -j8
```

### Incremental Build Times

```
Full rebuild:       ~60s (10+ cores)
Single file change: ~5s  (incremental)

Breakdown:
  - Tensor module: ~3s per file
  - Inference: ~2s
  - Bindings: ~4s (includes all headers)
  - Linking: ~2s
```

### Reducing Build Time

1. **Modular tensor implementation**: Change one file, rebuild only that file
2. **Precompiled headers**: Could add for common headers (not implemented)
3. **ccache**: Cache compilation results (optional)

```bash
# Install ccache
sudo apt install ccache

# Configure CMake to use it
cmake -DCMAKE_CXX_COMPILER_LAUNCHER=ccache ..
```

---

## Summary

The build system provides:

✅ **CMake-based**: Standard, portable build configuration
✅ **Modular sources**: Separate compilation units for fast rebuilds
✅ **Critical safety**: Lua compiled as C++ for exception safety
✅ **Cross-platform**: Supports x86_64, RISC-V, ARM
✅ **Optimized**: Release builds with -O3/-Os

⚠️ **Critical requirements**:
- Lua MUST be compiled with `-x c++`
- OpenCV 4.x required
- ONNX Runtime 1.12.0+
- C++17 compiler
