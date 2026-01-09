# Architecture Documentation Index

## Overview

This directory contains the complete architecture documentation for the model_infer project, describing the current system design, implementation details, and operational characteristics.

## Document Structure

### [01. System Overview](01-system-overview.md)
- Architecture layers (3-layer hybrid C++/Lua)
- Directory structure
- Module organization
- Design philosophy

### [02. Core Modules](02-core-modules.md)
- `lua_cv`: Computer Vision (Image, PreprocessRegistry)
- `tensor::Tensor`: N-dimensional arrays
- `inference::OnnxSession`: ONNX Runtime wrapper
- `lua_nn`: Neural Network bindings
- `lua_utils`: Utility functions

### [03. Data Flow](03-data-flow.md)
- Application entry point (main.cpp)
- Inference execution pipeline
- Dual preprocessing paths (C++ vs Lua)
- Complete pipeline visualization
- Video and image processing flows

### [04. Memory Management](04-memory-management.md)
- Shared ownership model
- CpuMemory lifecycle
- Zero-copy external data wrapping
- Storage sharing across tensor views

### [05. Lua Integration](05-lua-integration.md)
- Binding architecture (LuaIntf)
- Lua C API encapsulation policy
- Data conversion (C++ ↔ Lua)
- Type safety and exception handling

### [06. Build System](06-build-system.md)
- CMake structure
- Critical build configuration
- Lua compiled as C++ (exception safety)
- Dependency management

### [07. Testing Framework](07-testing-framework.md)
- Test organization (14 suites)
- Test execution flow
- Test utilities and helpers
- Coverage and validation

### [08. Performance Characteristics](08-performance.md)
- Operation costs analysis
- YOLO11n benchmark results
- Timing breakdown by component
- Performance optimization notes

### [09. Configuration](09-configuration.md)
- ONNX Runtime configuration
- OpenCV settings
- Thread configuration
- Optimization levels

---

## Quick Reference

**Target Platform**: Embedded Linux (RISC-V/ARM, <256MB RAM)

**Key Design Principles**:
- C++ for performance-critical operations
- Lua for business logic and flexibility
- Zero-copy tensor views for efficiency
- Device abstraction for future NPU/TPU support

**Performance Baseline**:
- YOLO11n (640×640): ~200ms per frame
- ONNX inference: ~100ms (50% of total time)
- Postprocessing: ~4.5ms (Tensor API)

---

## Document Purpose

These documents describe the **current architecture** as implemented. They focus on:
- What the system does and how it works
- Design decisions and their rationale
- Data flow patterns and interactions
- Performance characteristics

For implementation guides, code review, and development guidelines, see `CLAUDE.md` in the project root.
