# 5. Lua Integration

This document describes how C++ and Lua interact through the binding layer.

---

## 5.1 Binding Architecture

### LuaIntf Library

The system uses **LuaIntf** for type-safe C++/Lua bindings:

```
C++ Side                           Lua Side
┌──────────────────┐              ┌──────────────────┐
│ tensor::Tensor   │              │ nn.Tensor        │
│   .slice()       │◄─────────────┤   :slice()       │
│   .transpose()   │  LuaIntf     │   :transpose()   │
│   .contiguous()  │  binding     │   :contiguous()  │
└──────────────────┘              └──────────────────┘

Registration:
  LuaBinding(L).beginClass<Tensor>("Tensor")
      .addFunction("slice", &Tensor::slice)
      .addFunction("transpose", &Tensor::transpose)
      ...
  .endClass();

  LuaBinding(L).beginModule("nn")
      .addFactory("Tensor", ...)
  .endModule("nn");
```

### Binding Layers

```
┌────────────────────────────────────────────────┐
│ Layer 3: Lua Scripts                           │
│   local t = nn.Tensor.new(...)                 │
│   t:slice(0, 0, 10)                            │
└────────────────────────────────────────────────┘
                   ↓ ↑
┌────────────────────────────────────────────────┐
│ LuaIntf (Type-safe wrapper)                    │
│   - Argument validation                        │
│   - Type conversion                            │
│   - Exception handling                         │
└────────────────────────────────────────────────┘
                   ↓ ↑
┌────────────────────────────────────────────────┐
│ Lua C API                                      │
│   lua_State*, stack operations                 │
└────────────────────────────────────────────────┘
                   ↓ ↑
┌────────────────────────────────────────────────┐
│ Layer 1: C++ Implementation                    │
│   tensor::Tensor::slice(...)                   │
└────────────────────────────────────────────────┘
```

---

## 5.2 Lua C API Encapsulation Policy

### Critical Design Rule

**All Lua interaction MUST go through LuaIntf**, not raw Lua C API.

```cpp
// ❌ FORBIDDEN in src/ directory:
lua_pushnumber(L, value);
lua_tonumber(L, index);
luaL_checktype(L, index, LUA_TTABLE);

// ✅ REQUIRED: Use LuaIntf wrappers
LuaRef::fromValue(L, value);
LuaRef ref(L, index);
double val = ref.toValue<double>();
```

### Rationale

| Aspect | Raw Lua C API | LuaIntf |
|--------|--------------|---------|
| **Type safety** | Runtime errors | Compile-time checks |
| **Exception safety** | Manual cleanup | RAII across boundary |
| **Maintainability** | Scattered Lua code | Centralized in bindings/ |
| **Refactoring** | Error-prone | Safe (compiler helps) |

### Allowed Usage

```
src/
  ├── modules/
  │   ├── lua_cv.cpp         ✅ Uses LuaIntf only
  │   ├── lua_nn.cpp         ✅ Uses LuaIntf only
  │   └── lua_utils.cpp      ✅ Uses LuaIntf only
  │
  └── bindings/
      └── register_modules.cpp  ✅ Uses LuaIntf only

lua-intf-ex/                  ✅ Lua C API allowed here
  └── LuaIntf/                   (library implementation)
```

---

## 5.3 Data Conversion Patterns

### C++ → Lua Conversion

#### 1. C++ Objects (Userdata)

```cpp
// Tensor object
LuaBinding(L).beginClass<Tensor>("Tensor")
    .addConstructor<std::vector<int64_t>>()
    .addFunction("slice", &Tensor::slice)
    ...
.endClass();

// Lua receives:
local t = nn.Tensor.new({3, 4, 5})  -- userdata wrapping Tensor*
```

#### 2. Primitive Types

```cpp
// Numbers
LuaRef::fromValue(L, 42);           // int → Lua number
LuaRef::fromValue(L, 3.14);         // double → Lua number

// Strings
LuaRef::fromValue(L, "hello");      // const char* → Lua string
LuaRef::fromValue(L, std::string("world"));  // std::string → Lua string

// Booleans
LuaRef::fromValue(L, true);         // bool → Lua boolean
```

#### 3. Tables (Nested Structures)

```cpp
// Create Lua table from C++ vector
std::vector<std::vector<float>> data = {{1,2}, {3,4}, {5,6}};

LuaRef table = LuaRef::createTable(L);
for (size_t i = 0; i < data.size(); ++i) {
    LuaRef row = LuaRef::createTable(L);
    for (size_t j = 0; j < data[i].size(); ++j) {
        row[j+1] = data[i][j];  // Lua uses 1-based indexing
    }
    table[i+1] = row;
}
return table;

// Lua receives:
-- { {1,2}, {3,4}, {5,6} }
```

#### 4. Named Tables (Dictionaries)

```cpp
// Create metadata table
LuaRef meta = LuaRef::createTable(L);
meta["scale"] = 0.5;
meta["pad_x"] = 10;
meta["pad_y"] = 20;
meta["ori_w"] = 1920;
meta["ori_h"] = 1080;

// Lua receives:
-- {scale=0.5, pad_x=10, pad_y=20, ori_w=1920, ori_h=1080}
```

### Lua → C++ Conversion

#### 1. Extract Objects

```cpp
// From function argument
LuaRef tensorRef(L, 1);  // First argument (1-based)
Tensor* tensor = tensorRef.toValue<Tensor*>();

// From Lua global
LuaRef globalRef = LuaRef::fromGlobal(L, "my_tensor");
Tensor* tensor = globalRef.toValue<Tensor*>();
```

#### 2. Extract Primitives

```cpp
LuaRef ref(L, 2);  // Second argument

int i = ref.toValue<int>();
double d = ref.toValue<double>();
std::string s = ref.toValue<std::string>();
bool b = ref.toValue<bool>();
```

#### 3. Extract Tables

```cpp
// Lua: {1, 2, 3, 4, 5}
LuaRef tableRef(L, 1);
std::vector<int> vec;
for (int i = 1; i <= tableRef.length(); ++i) {  // Lua 1-based
    vec.push_back(tableRef[i].toValue<int>());
}

// Lua: {{1,2}, {3,4}}
LuaRef nestedRef(L, 1);
std::vector<std::vector<float>> matrix;
for (int i = 1; i <= nestedRef.length(); ++i) {
    LuaRef row = nestedRef[i];
    std::vector<float> rowVec;
    for (int j = 1; j <= row.length(); ++j) {
        rowVec.push_back(row[j].toValue<float>());
    }
    matrix.push_back(rowVec);
}
```

#### 4. Extract Named Fields

```cpp
// Lua: {type="letterbox", input_size={640,640}, stride=32}
LuaRef config(L, 1);

std::string type = config["type"].toValue<std::string>();
int stride = config["stride"].toValue<int>();

LuaRef sizeRef = config["input_size"];
int h = sizeRef[1].toValue<int>();  // Lua 1-based
int w = sizeRef[2].toValue<int>();
```

---

## 5.4 Function Binding Patterns

### 1. Member Functions

```cpp
// C++ class method
class Tensor {
public:
    Tensor slice(int dim, int start, int end);
};

// Binding
LuaBinding(L).beginClass<Tensor>("Tensor")
    .addFunction("slice", &Tensor::slice)
.endClass();

// Lua usage
local view = tensor:slice(0, 0, 10)
```

### 2. Static Functions

```cpp
// C++ static method
class Image {
public:
    static Image load(const std::string& path);
};

// Binding
LuaBinding(L).beginClass<Image>("Image")
    .addStaticFunction("load", &Image::load)
.endClass();

// Lua usage
local img = cv.Image.load("image.jpg")
```

### 3. Free Functions (Module-level)

```cpp
// C++ free function
std::vector<int> nms(const LuaRef& boxes, float iou_threshold);

// Binding
LuaBinding(L).beginModule("utils")
    .addFunction("nms", nms)
.endModule("utils");

// Lua usage
local keep = utils.nms(boxes, 0.45)
```

### 4. Functions Returning Multiple Values

```cpp
// C++ function returning struct
struct PreprocessResult {
    Tensor tensor;
    LuaRef meta;
};

PreprocessResult preprocess(Image& img, const LuaRef& config);

// Binding (LuaIntf automatically unpacks struct fields)
LuaBinding(L).beginModule("cv")
    .addFunction("preprocess", preprocess)
.endModule("cv");

// Lua usage
local tensor, meta = cv.preprocess(img, config)
```

---

## 5.5 Exception Handling Across Boundary

### C++ Exception → Lua Error

```cpp
// C++ code
Tensor Tensor::slice(int dim, int start, int end) {
    if (dim >= ndim()) {
        throw std::runtime_error("Dimension out of range");
    }
    // ...
}

// LuaIntf automatically converts exception to Lua error
```

```lua
-- Lua code
local status, result = pcall(function()
    return tensor:slice(10, 0, 5)  -- dim=10 out of range
end)

if not status then
    print("Error:", result)  -- "Dimension out of range"
end
```

### Lua Error → C++ Exception

```cpp
// Call Lua function from C++
LuaRef func = ...;

try {
    LuaRef result = func.call<LuaRef>(arg1, arg2);
} catch (const LuaException& e) {
    // Lua error caught as C++ exception
    std::cerr << "Lua error: " << e.what() << std::endl;
}
```

---

## 5.6 Object Lifetime Management

### C++ Object Owned by Lua

```
C++ creates object:
  tensor = Tensor({3, 4, 5})
  │
  └─→ Passed to Lua:
        LuaRef::fromValue(L, tensor)
        │
        ├─→ Tensor copied into Lua-managed userdata
        ├─→ Lua now owns the copy
        └─→ Original C++ tensor destroyed
              (shared_ptr refcount maintained by Lua copy)

Lua holds reference:
  local t = some_function()
  -- Tensor kept alive by Lua GC

Lua releases:
  t = nil
  collectgarbage("collect")
  │
  └─→ Userdata finalized
        └─→ Tensor destructor called
              └─→ shared_ptr<DeviceBuffer> refcount → 0
                    └─→ Memory freed
```

### C++ Object Borrowed by Lua (Reference)

```cpp
// Not commonly used, but possible:
Tensor tensor({3, 4, 5});

// Push pointer (Lua does NOT own)
LuaBinding(L).pushValue(&tensor);

// DANGER: tensor must outlive Lua's use of it!
// If tensor goes out of scope → dangling pointer in Lua
```

**Current system**: All objects passed to Lua are **owned** by Lua (copied via smart pointers).

---

## 5.7 Indexing Convention

### Lua 1-Based vs C++ 0-Based

| Context | Indexing |
|---------|----------|
| Lua tables | 1-based: `table[1]` = first element |
| C++ vectors | 0-based: `vec[0]` = first element |
| Tensor dimensions | 0-based: `tensor.slice(0, ...)` = first dim |
| Tensor argmax result | 0-based class IDs stored, but in 1-based Lua table |

**Example: argmax confusion**

```cpp
// C++ argmax returns 0-based indices
std::vector<int> indices = {0, 5, 2, 0, 1};  // class IDs

// Wrapped in Lua table (1-based container, 0-based values)
LuaRef result = LuaRef::createTable(L);
for (size_t i = 0; i < indices.size(); ++i) {
    result[i+1] = indices[i];  // Lua index i+1, value is 0-based
}
```

```lua
-- Lua receives: {0, 5, 2, 0, 1}
local class_ids = result

-- Access:
local first_class = class_ids[1]  -- 0 (C++ class ID, 0-based)
local label = labels[first_class + 1]  -- Need +1 for Lua label table
```

---

## Summary

The Lua integration architecture provides:

✅ **Type safety**: LuaIntf compile-time checks
✅ **Exception safety**: RAII and automatic conversion
✅ **Clean separation**: Bindings isolated in bindings/ directory
✅ **Easy maintenance**: Single point of Lua interaction
✅ **Automatic lifetime**: Lua GC manages C++ objects

⚠️ **Key considerations**:
- All Lua interaction through LuaIntf (not raw Lua C API)
- Mind 1-based (Lua) vs 0-based (C++) indexing
- Objects passed to Lua are owned by Lua GC
- Exception handling crosses language boundary automatically
