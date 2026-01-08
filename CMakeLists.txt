cmake_minimum_required(VERSION 3.18)
project(ModelInfer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ==========================================================
# 1. Build Lua as C++ (Critical for exception safety)
# ==========================================================
file(GLOB LUA_SRC "lua/*.c")
# Exclude standalone interpreter and compiler
list(REMOVE_ITEM LUA_SRC 
    "${CMAKE_CURRENT_SOURCE_DIR}/lua/lua.c" 
    "${CMAKE_CURRENT_SOURCE_DIR}/lua/luac.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/lua/onelua.c"
)

add_library(lua STATIC ${LUA_SRC})
target_compile_options(lua PRIVATE -x c++ -O3 -Wall -DLUA_USE_POSIX)
target_include_directories(lua PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/lua")

# ==========================================================
# 2. LuaIntf
# ==========================================================
# LuaIntf is header-only, but we need to make sure it finds our Lua
add_library(LuaIntf INTERFACE)
target_include_directories(LuaIntf INTERFACE 
    "${CMAKE_CURRENT_SOURCE_DIR}/lua-intf-ex/src/include"
)
target_link_libraries(LuaIntf INTERFACE lua)

# ==========================================================
# 3. Dependencies
# ==========================================================
find_package(OpenCV 4.6.0 REQUIRED)

# ONNX Runtime
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-prebuilt")
set(ONNXRUNTIME_INCLUDE_DIRS "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIRS}"
)

# ==========================================================
# 4. Main Application (Lua Runner)
# ==========================================================
# 收集模块源文件
file(GLOB MODULE_SOURCES "src/modules/*.cpp")
file(GLOB TENSOR_SOURCES "src/modules/tensor/*.cpp")
file(GLOB BINDING_SOURCES "src/bindings/*.cpp")
file(GLOB INFERENCE_SOURCES "src/inference/*.cpp")

add_executable(lua_runner
    src/main.cpp
    ${MODULE_SOURCES}
    ${TENSOR_SOURCES}
    ${BINDING_SOURCES}
    ${INFERENCE_SOURCES}
)

target_link_libraries(lua_runner PRIVATE
    LuaIntf
    lua
    ${OpenCV_LIBS}
    onnxruntime
)

target_include_directories(lua_runner PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/modules"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/modules/tensor"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/bindings"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils"
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

# Optional: Link dl library for Linux
if(UNIX AND NOT APPLE)
    target_link_libraries(lua_runner PRIVATE dl)
endif()

# 确保运行时能找到libonnxruntime.so
set_target_properties(lua_runner PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)

# ==========================================================
# 5. Pure C++ Inference App
# ==========================================================
add_executable(cpp_infer
    src/cpp_main.cpp
    ${INFERENCE_SOURCES}
)

target_link_libraries(cpp_infer PRIVATE
    ${OpenCV_LIBS}
    onnxruntime
)

target_include_directories(cpp_infer PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIRS}
)

if(UNIX AND NOT APPLE)
    target_link_libraries(cpp_infer PRIVATE dl)
endif()

set_target_properties(cpp_infer PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)


