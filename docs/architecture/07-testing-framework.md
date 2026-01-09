# 7. Testing Framework

This document describes the Lua-based testing framework for validating tensor operations and utilities.

---

## 7.1 Test Organization

### Test Suite Structure

```
tests/
  ├── run_all_tests.lua           # Main test runner
  ├── test_helpers.lua            # Assertion and utility functions
  │
  ├── test_tensor_core.lua        # Basic tensor operations
  ├── test_tensor_shape.lua       # Shape manipulation
  ├── test_tensor_math.lua        # Arithmetic operations
  ├── test_tensor_activation.lua  # Activation functions
  ├── test_tensor_compare.lua     # Comparison operations
  ├── test_tensor_reduction.lua   # Reduction operations
  ├── test_tensor_select.lua      # Selection and indexing
  ├── test_tensor_device.lua      # Device operations
  ├── test_tensor_advanced.lua    # Advanced operations
  │
  ├── test_image.lua              # Image operations
  ├── test_preprocessing.lua      # Preprocessing functions
  ├── test_utils.lua              # Utility functions (NMS, etc.)
  └── test_inference.lua          # End-to-end inference

Total: 14 test suites
```

---

## 7.2 Running Tests

### Execute All Tests

```bash
./build/lua_runner tests/run_all_tests.lua
```

**Output**:
```
========== 运行Lua测试脚本 ==========
加载脚本: tests/run_all_tests.lua
注册模块...
执行测试脚本...

Running test_tensor_core.lua...
  ✓ test_tensor_creation
  ✓ test_tensor_shape
  ✓ test_tensor_access
  ...

Running test_tensor_shape.lua...
  ✓ test_slice
  ✓ test_transpose
  ✓ test_squeeze
  ...

=== Test Summary ===
Passed: 87 / 87
Failed: 0
✓ All tests passed!
```

### Execute Single Suite

```bash
./build/lua_runner tests/test_tensor_core.lua
```

---

## 7.3 Test Framework Implementation

### 7.3.1 Test Runner

```lua
-- tests/run_all_tests.lua
local test_helpers = require("tests.test_helpers")

local test_suites = {
    "tests.test_tensor_core",
    "tests.test_tensor_shape",
    "tests.test_tensor_math",
    -- ... more suites
}

for _, suite_name in ipairs(test_suites) do
    print("Running " .. suite_name .. "...")
    local suite = require(suite_name)

    -- Execute test suite
    suite.run_tests()
end

-- Print summary
test_helpers.print_summary()
```

### 7.3.2 Test Helper Utilities

```lua
-- tests/test_helpers.lua
local M = {}

-- Test state
M.passed = 0
M.failed = 0
M.current_test = ""

-- Assertion functions
function M.assert_eq(actual, expected, message)
    if actual ~= expected then
        error(string.format(
            "%s: expected %s, got %s",
            message or "Assertion failed",
            tostring(expected),
            tostring(actual)
        ))
    end
end

function M.assert_near(actual, expected, tolerance, message)
    tolerance = tolerance or 1e-5
    if math.abs(actual - expected) > tolerance then
        error(string.format(
            "%s: expected %f, got %f (tolerance %f)",
            message or "Assertion failed",
            expected, actual, tolerance
        ))
    end
end

function M.assert_tensor_eq(t1, t2, tolerance, message)
    -- Compare shapes
    local shape1 = t1:shape()
    local shape2 = t2:shape()
    M.assert_eq(#shape1, #shape2, "Shape dimension mismatch")

    for i = 1, #shape1 do
        M.assert_eq(shape1[i], shape2[i], "Shape mismatch at dim " .. i)
    end

    -- Compare values
    local data1 = t1:to_table()
    local data2 = t2:to_table()

    M.compare_tables(data1, data2, tolerance)
end

function M.compare_tables(t1, t2, tolerance)
    -- Recursive comparison with tolerance
    -- (implementation details)
end

-- Test execution wrapper
function M.run_test(name, test_func)
    M.current_test = name
    local status, err = pcall(test_func)

    if status then
        M.passed = M.passed + 1
        print("  ✓ " .. name)
    else
        M.failed = M.failed + 1
        print("  ✗ " .. name)
        print("    Error: " .. tostring(err))
    end
end

-- Summary
function M.print_summary()
    print("\n=== Test Summary ===")
    print(string.format("Passed: %d", M.passed))
    print(string.format("Failed: %d", M.failed))

    if M.failed == 0 then
        print("✓ All tests passed!")
    else
        print("✗ Some tests failed")
        os.exit(1)
    end
end

return M
```

---

## 7.4 Test Suite Example

### Basic Structure

```lua
-- tests/test_tensor_shape.lua
local helpers = require("tests.test_helpers")
local nn = lua_nn

local M = {}

function M.test_slice()
    local t = nn.Tensor.new({4, 5})
    -- Fill with test data

    local sliced = t:slice(0, 1, 3)

    helpers.assert_eq(#sliced:shape(), 2, "Sliced should be 2D")
    helpers.assert_eq(sliced:shape()[1], 2, "First dim should be 2")
    helpers.assert_eq(sliced:shape()[2], 5, "Second dim should be 5")
end

function M.test_transpose()
    local t = nn.Tensor.new({3, 4})
    local transposed = t:transpose(0, 1)

    helpers.assert_eq(transposed:shape()[1], 4, "First dim should be 4")
    helpers.assert_eq(transposed:shape()[2], 3, "Second dim should be 3")

    -- Verify data layout
    -- (value checks)
end

function M.test_squeeze()
    local t = nn.Tensor.new({1, 3, 1, 4})
    local squeezed = t:squeeze(0)

    helpers.assert_eq(#squeezed:shape(), 3, "Should be 3D after squeeze")
    helpers.assert_eq(squeezed:shape()[1], 3, "First dim")
end

function M.run_tests()
    helpers.run_test("test_slice", M.test_slice)
    helpers.run_test("test_transpose", M.test_transpose)
    helpers.run_test("test_squeeze", M.test_squeeze)
    -- ... more tests
end

return M
```

---

## 7.5 Test Coverage

### Tensor Operations

**Core operations** (`test_tensor_core.lua`):
- Construction (from shape, from data)
- Shape query
- Data access (at, data pointer)
- Cloning

**Shape operations** (`test_tensor_shape.lua`):
- slice, transpose, squeeze, unsqueeze
- view, reshape, permute
- Contiguity checking

**Math operations** (`test_tensor_math.lua`):
- Element-wise: add, sub, mul, div
- In-place variants: add_, sub_, mul_, div_
- Scalar operations

**Activation functions** (`test_tensor_activation.lua`):
- sigmoid, sigmoid_
- relu, relu_ (if implemented)

**Reduction operations** (`test_tensor_reduction.lua`):
- max, argmax, max_with_argmax
- sum, mean

**Selection operations** (`test_tensor_select.lua`):
- where_indices
- index_select
- extract_columns

**Advanced operations** (`test_tensor_advanced.lua`):
- gather, concat, split

### Utility Functions

**Image operations** (`test_image.lua`):
- load, save
- resize, pad
- to_tensor conversion

**Preprocessing** (`test_preprocessing.lua`):
- letterbox
- resize_center_crop
- Coordinate scaling

**Utils** (`test_utils.lua`):
- NMS (Non-Maximum Suppression)
- Box format conversion
- IoU calculation

### Integration Tests

**End-to-end inference** (`test_inference.lua`):
- Model loading
- Preprocessing
- Inference execution
- Postprocessing
- Result validation

---

## 7.6 Test Patterns

### Pattern 1: Value Verification

```lua
function test_add()
    local a = nn.Tensor.new({2, 3})
    local b = nn.Tensor.new({2, 3})

    -- Fill with known values
    -- a = [[1,2,3], [4,5,6]]
    -- b = [[10,20,30], [40,50,60]]

    local c = a:add(b)

    -- Verify result
    local expected = nn.Tensor.new({2, 3})
    -- expected = [[11,22,33], [44,55,66]]

    helpers.assert_tensor_eq(c, expected, 1e-5, "Add result")
end
```

### Pattern 2: Shape Verification

```lua
function test_reshape()
    local t = nn.Tensor.new({2, 3, 4})
    local reshaped = t:reshape({6, 4})

    helpers.assert_eq(#reshaped:shape(), 2, "Should be 2D")
    helpers.assert_eq(reshaped:shape()[1], 6, "First dim")
    helpers.assert_eq(reshaped:shape()[2], 4, "Second dim")
end
```

### Pattern 3: Error Testing

```lua
function test_invalid_slice()
    local t = nn.Tensor.new({3, 4})

    -- Should throw error
    local status, err = pcall(function()
        t:slice(5, 0, 2)  -- Invalid dimension
    end)

    helpers.assert_eq(status, false, "Should throw error")
end
```

### Pattern 4: Zero-Copy Verification

```lua
function test_slice_shares_storage()
    local t = nn.Tensor.new({4, 5})
    -- Fill with data

    local view = t:slice(0, 0, 2)

    -- Modify original
    -- (via data pointer or in-place op)

    -- Verify view sees the change
    -- (confirms storage is shared)
end
```

---

## 7.7 Running Tests in CI/CD

### Basic CI Script

```bash
#!/bin/bash
# ci_test.sh

set -e  # Exit on error

echo "Building project..."
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo "Running tests..."
./lua_runner ../tests/run_all_tests.lua

echo "Tests passed!"
```

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Test

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Install dependencies
        run: |
          sudo apt update
          sudo apt install -y libopencv-dev lua5.3 liblua5.3-dev

      - name: Build
        run: |
          mkdir build && cd build
          cmake ..
          make -j$(nproc)

      - name: Run tests
        run: |
          ./build/lua_runner tests/run_all_tests.lua
```

---

## Summary

The testing framework provides:

✅ **Comprehensive coverage**: 14 test suites covering all modules
✅ **Easy to run**: Single command for all tests
✅ **Clear output**: Pass/fail with error details
✅ **Lua-based**: Tests written in same language as scripts
✅ **Modular**: Each suite tests specific functionality

**Test categories**:
- Unit tests: Individual tensor operations
- Integration tests: Preprocessing, inference, postprocessing
- Regression tests: Ensure behavior remains consistent

**Best practices**:
- Write tests for new features
- Run tests before committing
- Add regression tests for bug fixes
- Verify edge cases (empty tensors, invalid dimensions)
