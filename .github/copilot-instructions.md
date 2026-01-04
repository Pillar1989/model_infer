# GitHub Copilot Instructions for Model Inference Project

This repository implements a high-performance model inference engine using Lua scripting with C++ bindings, leveraging `lua-intf` and `onnxruntime`.

## 🏗 Project Architecture

- **Build System**: CMake-based build at the project root, managing submodules (`lua`, `lua-intf`) and compilation.
- **Core**: Lua 5.x compiled as C++ to ensure exception safety across language boundaries.
- **Bindings**: `lua-intf` (header-only C++11 library) binds C++ classes/functions to Lua.
- **Inference**: `onnxruntime` (ORT) is the backend for executing ONNX models.
- **Logic**: Lua scripts (e.g., `scripts/yolov5_detector.lua`) control the inference pipeline (preprocessing -> inference -> postprocessing).
- **Modules**:
  - `lua_cv`: Computer vision primitives (Image, resize, padding) implemented using **OpenCV 4.x**.
  - `lua_nn`/`lua_utils`: Post-processing (Tensor parsing, NMS, Box handling).

## 🛠 Build & Development Workflow

### 1. Build System (CMake)
The project uses a unified CMake build system at the root level. It automatically compiles Lua as C++ (critical for exception safety) and links `lua-intf`.

```bash
mkdir build && cd build
cmake ..
make
```

### 2. Running Scripts
The main executable `model_inference` is generated in the build directory.
```bash
# Example execution
./build/model_inference scripts/yolov5_detector.lua models/yolov5.onnx images/zidane.jpg
```

### 3. Submodules
Ensure submodules (`lua`, `lua-intf`, `onnxruntime`) are initialized:
```bash
git submodule update --init --recursive
```

## 🧩 Coding Conventions

### C++ (Host/Modules)
- **LuaIntf**: Use `LuaBinding(L).beginModule(...)` to expose APIs.
- **Memory**: Use `LuaRef` for holding Lua objects in C++.
- **Exceptions**: Throw `LuaException` or standard exceptions; `lua-intf` translates them to Lua errors.
- **Headers**: `lua-intf` is header-only; include `LuaIntf.h`.

### Lua (Scripts)
- **Modules**: Access C++ functionality via required modules (e.g., `local cv = require "lua_cv"`).
- **Types**: C++ classes (like `Image`, `Tensor`) are exposed as Userdata but behave like Lua objects (methods, properties).
- **Indexing**: Remember Lua uses 1-based indexing.

## ⚠️ Critical Implementation Details

- **Computer Vision**: All CV operations (image loading, resizing, color conversion) MUST be implemented using **OpenCV 4.x**. The system provides OpenCV 4.6.0.
- **Stubbed Implementations**: Some C++ modules currently contain stub/mock implementations. When implementing real logic, replace these stubs.
- **ONNX Runtime**: Integration requires binding `Ort::Session` and `Ort::Value` to Lua. Ensure tensor data is passed efficiently (zero-copy if possible) between C++ and Lua.
- **Module Naming**: Be aware of potential naming discrepancies between C++ registration (`luaopen_CVLib`) and Lua usage (`require "lua_cv"`). Ensure consistency in `lua_setglobal` or `package.preload`.

## 🔍 Key Files
- `lua/onelua.c`: Single-file Lua build (reference).
- `lua-intf/src/include/LuaIntf.h`: Main binding header.
- `lua-intf/tests/src/cv_module.cpp`: Example CV module implementation.
- `scripts/yolov5_detector.lua`: Main inference script example.
