# Model Inference Engine

A high-performance, flexible model inference engine combining the speed of C++ with the scripting flexibility of Lua. Built with **ONNX Runtime**, **OpenCV**, and **LuaIntf**.

## 🚀 Features

- **Hybrid Architecture**: Core inference and heavy lifting in C++, business logic and pre/post-processing in Lua.
- **High Performance**: Zero-copy tensor passing where possible, optimized OpenCV operations.
- **Flexible**: Modify inference pipelines (preprocessing, NMS, etc.) without recompiling C++.
- **Robust**: Handles dynamic input shapes (Auto-padding), Float16/Float32 type mismatch automatically.
- **Comparison Tool**: Includes a pure C++ implementation (`cpp_infer`) to benchmark against the Lua-scripted engine.

## 🛠️ Prerequisites

- **C++ Compiler**: Supporting C++20.
- **CMake**: Version 3.18 or higher.
- **OpenCV**: Version 4.x installed on the system.
- **ONNX Runtime**: Prebuilt binaries (see below).

## 📦 Setup & Build

### 1. Prepare ONNX Runtime
**⚠️ Important**: This project requires ONNX Runtime prebuilt binaries.
You must manually ensure the `onnxruntime-prebuilt` directory contains the `include` and `lib` folders.

1. Download the ONNX Runtime (Linux x64) release (compatible with your system).
2. Extract it into the project root and rename/link it to `onnxruntime-prebuilt`.

Structure should look like:
```
project_root/
├── onnxruntime-prebuilt/
│   ├── include/
│   └── lib/
│       └── libonnxruntime.so
├── src/
├── scripts/
...
```

### 2. Build
```bash
mkdir build && cd build
cmake ..
make
```

## 🏃 Usage

### Lua Inference Engine (Recommended)
The main engine uses Lua scripts to define the pipeline.

```bash
# Syntax
./model_infer <script_path> <model_path> <image_path> [show]

# Example
./model_infer ../scripts/yolov5_detector.lua ../models/yolov5n.onnx ../images/zidane.jpg

# With Visualization
./model_infer ../scripts/yolov5_detector.lua ../models/yolov5n.onnx ../images/zidane.jpg show
```

### Pure C++ Inference (Benchmark)
A hardcoded C++ implementation is provided for performance comparison.

```bash
# Syntax
./cpp_infer <model_path> <image_path> [show]

# Example
./cpp_infer ../models/yolov5n.onnx ../images/zidane.jpg
```

## � Benchmark Results

Comparison between Lua-scripted engine (`model_infer`) and pure C++ implementation (`cpp_infer`) on YOLOv5n.

| Metric | Lua Engine (`model_infer`) | C++ Engine (`cpp_infer`) | Difference |
| :--- | :--- | :--- | :--- |
| **Inference Time** (Real) | ~270 ms | ~240 ms | C++ is ~11% faster |
| **Memory Usage** (RSS) | ~150 MB | ~145 MB | C++ uses ~5 MB less |

*Note: Tested on Linux x64 AMD Ryzen 9 3900X 12-Core Processor. "Inference Time" includes initialization, image loading, preprocessing, inference, and postprocessing.*

## �📂 Project Structure

- `src/`: C++ source code.
  - `modules/`: Lua bindings for CV and NN operations.
  - `main.cpp`: Lua engine entry point.
  - `cpp_main.cpp`: Pure C++ implementation.
- `scripts/`: Lua scripts defining inference logic (e.g., YOLOv5).
- `lua/`: Lua 5.x source code (compiled as C++).
- `lua-intf/`: C++/Lua binding library.
- `onnxruntime-prebuilt/`: ONNX Runtime headers and libraries.
