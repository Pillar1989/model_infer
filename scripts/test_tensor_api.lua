-- Tensor API 完整测试脚本
-- 使用方式: ./build/test_tensor scripts/test_tensor_api.lua

local nn = lua_nn

print("========================================")
print("    Tensor API 完整测试")
print("========================================\n")

local test_count = 0
local pass_count = 0

local function test(name, fn)
    test_count = test_count + 1
    io.write(string.format("测试%d: %s ... ", test_count, name))
    local ok, err = pcall(fn)
    if ok then
        pass_count = pass_count + 1
        print("✓")
    else
        print("✗")
        print("  错误:", err)
    end
end

local function assert_eq(a, b, msg)
    if a ~= b then
        error(string.format("%s: expected %s, got %s", msg or "assertion failed", tostring(b), tostring(a)))
    end
end

local function assert_near(a, b, eps, msg)
    eps = eps or 1e-5
    if math.abs(a - b) > eps then
        error(string.format("%s: expected ~%s, got %s", msg or "assertion failed", tostring(b), tostring(a)))
    end
end

-- ========================================
-- Level 1: 基础属性和构造
-- ========================================
print("\n========== Level 1: 基础属性和构造 ==========\n")

test("基础构造 - 1D", function()
    local t = nn.Tensor.new({1, 2, 3, 4}, {4})
    assert_eq(t.ndim, 1, "ndim")
    assert_eq(t:size(), 4, "size")
    assert(t:is_contiguous(), "should be contiguous")
end)

test("基础构造 - 2D", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    assert_eq(t.ndim, 2, "ndim")
    assert_eq(t:size(), 6, "size")
    local shape = t:shape()
    assert_eq(shape[1], 2, "shape[1]")
    assert_eq(shape[2], 3, "shape[2]")
end)

test("基础构造 - 3D", function()
    local t = nn.Tensor.new({1,2,3,4,5,6,7,8,9,10,11,12}, {2, 2, 3})
    assert_eq(t.ndim, 3, "ndim")
    assert_eq(t:size(), 12, "size")
end)

test("strides属性", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local strides = t:strides()
    assert_eq(strides[1], 3, "stride[1]")
    assert_eq(strides[2], 1, "stride[2]")
end)

test("get/set元素访问", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    assert_eq(t:get(0), 1, "get(0)")
    assert_eq(t:get(5), 6, "get(5)")
    t:set(0, 100)
    assert_eq(t:get(0), 100, "set then get")
end)

test("to_table转换", function()
    local t = nn.Tensor.new({1, 2, 3, 4}, {4})
    local tbl = t:to_table()
    assert_eq(#tbl, 4, "table length")
    assert_eq(tbl[1], 1, "tbl[1]")
    assert_eq(tbl[4], 4, "tbl[4]")
end)

test("to_string输出", function()
    local t = nn.Tensor.new({1, 2, 3}, {3})
    local s = t:to_string(10)
    assert(s:find("Tensor"), "should contain 'Tensor'")
end)

test("Metamethod __len", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5}, {5})
    assert_eq(#t, 5, "__len")
end)

test("Metamethod __tostring", function()
    local t = nn.Tensor.new({1, 2}, {2})
    local s = tostring(t)
    assert(s:find("Tensor"), "__tostring")
end)

-- ========================================
-- Level 2: 形状操作
-- ========================================
print("\n========== Level 2: 形状操作 ==========\n")

test("slice - 基础切片", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4})
    local sliced = t:slice(0, 0, 2, 1)
    local shape = sliced:shape()
    assert_eq(shape[1], 2, "sliced rows")
    assert_eq(shape[2], 4, "sliced cols")
end)

test("slice - 带步长", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4})
    local sliced = t:slice(0, 0, 3, 2)  -- step=2
    local shape = sliced:shape()
    assert_eq(shape[1], 2, "sliced with step")
end)

test("select_dim", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local selected = t:select_dim(0, 1)  -- 选择第2行
    local shape = selected:shape()
    assert_eq(shape[1], 3, "selected dim shape")
end)

test("get_column", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local col = t:get_column(1)  -- 获取第2列
    assert_eq(col:size(), 2, "column size")
end)

test("slice_columns", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local cols = t:slice_columns(0, 2)  -- 前2列
    local shape = cols:shape()
    assert_eq(shape[2], 2, "sliced columns")
end)

test("reshape", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local reshaped = t:reshape({3, 2})
    local shape = reshaped:shape()
    assert_eq(shape[1], 3, "reshaped[1]")
    assert_eq(shape[2], 2, "reshaped[2]")
end)

test("reshape - 1D to 2D", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {6})
    local reshaped = t:reshape({2, 3})
    assert_eq(reshaped.ndim, 2, "ndim after reshape")
end)

test("transpose - 2D", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local transposed = t:transpose()
    local shape = transposed:shape()
    assert_eq(shape[1], 3, "transposed[1]")
    assert_eq(shape[2], 2, "transposed[2]")
end)

test("transpose_dims - 自定义维度", function()
    local t = nn.Tensor.new({1,2,3,4,5,6,7,8,9,10,11,12}, {2, 2, 3})
    local transposed = t:transpose_dims({2, 0, 1})
    local shape = transposed:shape()
    assert_eq(shape[1], 3, "transposed_dims[1]")
    assert_eq(shape[2], 2, "transposed_dims[2]")
    assert_eq(shape[3], 2, "transposed_dims[3]")
end)

test("squeeze - 移除所有单一维度", function()
    local t = nn.Tensor.new({1, 2, 3}, {1, 3, 1})
    local squeezed = t:squeeze(-1)  -- dim=-1 removes ALL singleton dims
    local shape = squeezed:shape()
    assert_eq(#shape, 1, "squeezed ndim")  -- {1,3,1} -> {3}
    assert_eq(shape[1], 3, "squeezed shape")
end)

test("squeeze - 指定维度", function()
    local t = nn.Tensor.new({1, 2, 3}, {1, 3, 1})
    local squeezed = t:squeeze(0)  -- only remove first dim
    local shape = squeezed:shape()
    assert_eq(#shape, 2, "squeezed ndim")  -- {1,3,1} -> {3,1}
end)

test("unsqueeze", function()
    local t = nn.Tensor.new({1, 2, 3}, {3})
    local unsqueezed = t:unsqueeze(0)
    local shape = unsqueezed:shape()
    assert_eq(shape[1], 1, "unsqueezed[1]")
    assert_eq(shape[2], 3, "unsqueezed[2]")
end)

-- ========================================
-- Level 2.5: Contiguous标记验证
-- ========================================
print("\n========== Level 2.5: Contiguous标记验证 ==========\n")

test("contiguous标记 - 初始tensor", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    assert(t:is_contiguous(), "initial tensor should be contiguous")
end)

test("contiguous标记 - slice全部行", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local sliced = t:slice(0, 0, 2, 1)  -- 全部行
    assert(sliced:is_contiguous(), "full slice should be contiguous")
end)

test("contiguous标记 - slice部分行", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6, 7, 8, 9}, {3, 3})
    local sliced = t:slice(0, 0, 2, 1)  -- 前2行 (不包含所有行)
    -- 当前实现：只有slice包含所有元素时才标记为连续
    assert_eq(sliced:is_contiguous(), false, "partial row slice is non-contiguous")
end)

test("contiguous标记 - slice部分列", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local sliced = t:slice(1, 0, 2, 1)  -- 前2列
    assert_eq(sliced:is_contiguous(), false, "partial column slice should NOT be contiguous")
end)

test("contiguous标记 - slice_columns", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local cols = t:slice_columns(0, 2)  -- 前2列
    assert_eq(cols:is_contiguous(), false, "slice_columns should NOT be contiguous")
end)

test("contiguous标记 - slice_columns全部列", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local cols = t:slice_columns(0, 3)  -- 全部3列
    assert(cols:is_contiguous(), "slice_columns all cols should be contiguous")
end)

test("contiguous标记 - transpose", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local transposed = t:transpose()
    assert_eq(transposed:is_contiguous(), false, "transpose should NOT be contiguous")
end)

test("contiguous标记 - select_dim", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local selected = t:select_dim(0, 1)  -- 选择第2行
    -- select_dim 实际上是 slice(dim, index, index+1) 然后 squeeze
    -- 结果是一个view，非连续
    assert_eq(selected:is_contiguous(), false, "select_dim creates view, non-contiguous")
end)

test("contiguous标记 - get_column", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local col = t:get_column(1)
    assert_eq(col:is_contiguous(), false, "get_column should NOT be contiguous")
end)

test("contiguous方法 - 已连续", function()
    local t = nn.Tensor.new({1, 2, 3, 4}, {4})
    local cont = t:contiguous()
    assert(cont:is_contiguous(), "contiguous() result should be contiguous")
    -- 应该返回同一个对象（不复制）
    assert_eq(cont:get(0), 1, "values preserved")
end)

test("contiguous方法 - 非连续", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local cols = t:slice_columns(1, 3)  -- 非连续
    assert_eq(cols:is_contiguous(), false, "before contiguous()")

    local cont = cols:contiguous()
    assert(cont:is_contiguous(), "after contiguous() should be contiguous")

    -- 验证数据正确性
    local row0 = cont:select_dim(0, 0):to_table()
    assert_eq(row0[1], 2, "row 0 col 0")
    assert_eq(row0[2], 3, "row 0 col 1")
end)

test("view操作组合 - slice后slice", function()
    local t = nn.Tensor.new({1,2,3,4,5,6,7,8,9,10,11,12}, {3, 4})
    local sliced1 = t:slice(0, 0, 3, 1)  -- 全部行
    local sliced2 = sliced1:slice(1, 1, 3, 1)  -- 中间2列
    assert_eq(sliced2:is_contiguous(), false, "double slice non-contiguous")

    -- 验证数据
    local val = sliced2:get(0)  -- row 0, col 1 (原始) = 2
    assert_eq(val, 2, "double slice value")
end)

test("view操作组合 - transpose后slice", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local transposed = t:transpose()  -- [3, 2]
    local sliced = transposed:slice(0, 0, 2, 1)  -- 前2行
    assert_eq(sliced:is_contiguous(), false, "transpose then slice")

    -- 验证形状和数据
    local shape = sliced:shape()
    assert_eq(shape[1], 2, "sliced shape[0]")
    assert_eq(shape[2], 2, "sliced shape[1]")
end)

-- ========================================
-- Level 3: 数学运算 - 标量
-- ========================================
print("\n========== Level 3: 数学运算 - 标量 ==========\n")

test("add标量", function()
    local t = nn.Tensor.new({1, 2, 3, 4}, {4})
    local result = t:add(10)
    assert_eq(result:get(0), 11, "add result[0]")
    assert_eq(result:get(3), 14, "add result[3]")
end)

test("sub标量", function()
    local t = nn.Tensor.new({10, 20, 30}, {3})
    local result = t:sub(5)
    assert_eq(result:get(0), 5, "sub result")
end)

test("mul标量", function()
    local t = nn.Tensor.new({1, 2, 3}, {3})
    local result = t:mul(3)
    assert_eq(result:get(0), 3, "mul result[0]")
    assert_eq(result:get(2), 9, "mul result[2]")
end)

test("div标量", function()
    local t = nn.Tensor.new({10, 20, 30}, {3})
    local result = t:div(10)
    assert_eq(result:get(0), 1, "div result[0]")
    assert_eq(result:get(2), 3, "div result[2]")
end)

test("Metamethod __add", function()
    local t = nn.Tensor.new({1, 2, 3}, {3})
    local result = t + 5
    assert_eq(result:get(0), 6, "__add result")
end)

test("Metamethod __sub", function()
    local t = nn.Tensor.new({10, 20}, {2})
    local result = t - 5
    assert_eq(result:get(0), 5, "__sub result")
end)

test("Metamethod __mul", function()
    local t = nn.Tensor.new({2, 3}, {2})
    local result = t * 4
    assert_eq(result:get(0), 8, "__mul result")
end)

test("Metamethod __div", function()
    local t = nn.Tensor.new({10, 20}, {2})
    local result = t / 2
    assert_eq(result:get(0), 5, "__div result")
end)

-- ========================================
-- Level 4: 数学运算 - Tensor
-- ========================================
print("\n========== Level 4: 数学运算 - Tensor ==========\n")

test("add_tensor", function()
    local t1 = nn.Tensor.new({1, 2, 3}, {3})
    local t2 = nn.Tensor.new({10, 20, 30}, {3})
    local result = t1:add_tensor(t2)
    assert_eq(result:get(0), 11, "add_tensor result[0]")
    assert_eq(result:get(2), 33, "add_tensor result[2]")
end)

test("sub_tensor", function()
    local t1 = nn.Tensor.new({10, 20, 30}, {3})
    local t2 = nn.Tensor.new({1, 2, 3}, {3})
    local result = t1:sub_tensor(t2)
    assert_eq(result:get(0), 9, "sub_tensor result")
end)

test("mul_tensor", function()
    local t1 = nn.Tensor.new({2, 3, 4}, {3})
    local t2 = nn.Tensor.new({5, 6, 7}, {3})
    local result = t1:mul_tensor(t2)
    assert_eq(result:get(0), 10, "mul_tensor result[0]")
    assert_eq(result:get(2), 28, "mul_tensor result[2]")
end)

test("div_tensor", function()
    local t1 = nn.Tensor.new({10, 20, 30}, {3})
    local t2 = nn.Tensor.new({2, 4, 5}, {3})
    local result = t1:div_tensor(t2)
    assert_eq(result:get(0), 5, "div_tensor result[0]")
    assert_eq(result:get(2), 6, "div_tensor result[2]")
end)

-- ========================================
-- Level 5: 归约操作
-- ========================================
print("\n========== Level 5: 归约操作 ==========\n")

test("sum - 全局", function()
    local t = nn.Tensor.new({1, 2, 3, 4}, {4})
    local result = t:sum(-1, false)
    assert_eq(result:get(0), 10, "sum all")
end)

test("sum - 沿维度", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local result = t:sum(1, true)
    local shape = result:shape()
    assert_eq(shape[1], 2, "sum dim shape[1]")
    assert_eq(shape[2], 1, "sum dim shape[2]")
end)

test("mean - 全局", function()
    local t = nn.Tensor.new({2, 4, 6, 8}, {4})
    local result = t:mean(-1, false)
    assert_eq(result:get(0), 5, "mean all")
end)

test("max - 全局", function()
    local t = nn.Tensor.new({3, 1, 4, 1, 5, 9}, {6})
    local result = t:max(-1, false)
    assert_eq(result:get(0), 9, "max all")
end)

test("min - 全局", function()
    local t = nn.Tensor.new({3, 1, 4, 1, 5, 9}, {6})
    local result = t:min(-1, false)
    assert_eq(result:get(0), 1, "min all")
end)

test("argmax", function()
    local t = nn.Tensor.new({3, 1, 4, 1, 5, 9, 2, 6}, {8})
    local max_idx = t:argmax(-1)
    assert_eq(max_idx, 5, "argmax index")  -- 9 at index 5
end)

test("argmin", function()
    local t = nn.Tensor.new({3, 1, 4, 1, 5, 9, 2, 6}, {8})
    local min_idx = t:argmin(-1)
    assert_eq(min_idx, 1, "argmin index")  -- 1 at index 1
end)

-- ========================================
-- Level 6: 激活函数
-- ========================================
print("\n========== Level 6: 激活函数 ==========\n")

test("sigmoid", function()
    local t = nn.Tensor.new({0}, {1})
    local result = t:sigmoid()
    assert_near(result:get(0), 0.5, 1e-5, "sigmoid(0)")
end)

test("sigmoid - 多值", function()
    local t = nn.Tensor.new({-1, 0, 1, 2}, {4})
    local result = t:sigmoid()
    assert_near(result:get(1), 0.5, 1e-5, "sigmoid middle")
    assert(result:get(0) < 0.5, "sigmoid negative < 0.5")
    assert(result:get(2) > 0.5, "sigmoid positive > 0.5")
end)

test("softmax", function()
    local t = nn.Tensor.new({1, 2, 3}, {3})
    local result = t:softmax(-1)
    -- softmax输出和应为1
    local sum = result:get(0) + result:get(1) + result:get(2)
    assert_near(sum, 1.0, 1e-5, "softmax sum")
end)

test("exp", function()
    local t = nn.Tensor.new({0, 1}, {2})
    local result = t:exp()
    assert_near(result:get(0), 1.0, 1e-5, "exp(0)")
    assert_near(result:get(1), 2.718281828, 1e-5, "exp(1)")
end)

test("log", function()
    local t = nn.Tensor.new({1, 2.718281828}, {2})
    local result = t:log()
    assert_near(result:get(0), 0.0, 1e-5, "log(1)")
    assert_near(result:get(1), 1.0, 1e-5, "log(e)")
end)

-- ========================================
-- Level 7: 比较操作
-- ========================================
print("\n========== Level 7: 比较操作 ==========\n")

test("gt (>)", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5}, {5})
    local mask = t:gt(3)
    assert_eq(mask:get(0), 0, "1 > 3")
    assert_eq(mask:get(2), 0, "3 > 3")
    assert_eq(mask:get(3), 1, "4 > 3")
    assert_eq(mask:get(4), 1, "5 > 3")
end)

test("lt (<)", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5}, {5})
    local mask = t:lt(3)
    assert_eq(mask:get(0), 1, "1 < 3")
    assert_eq(mask:get(1), 1, "2 < 3")
    assert_eq(mask:get(2), 0, "3 < 3")
end)

test("ge (>=)", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5}, {5})
    local mask = t:ge(3)
    assert_eq(mask:get(1), 0, "2 >= 3")
    assert_eq(mask:get(2), 1, "3 >= 3")
    assert_eq(mask:get(3), 1, "4 >= 3")
end)

test("le (<=)", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5}, {5})
    local mask = t:le(3)
    assert_eq(mask:get(0), 1, "1 <= 3")
    assert_eq(mask:get(2), 1, "3 <= 3")
    assert_eq(mask:get(3), 0, "4 <= 3")
end)

test("eq (==)", function()
    local t = nn.Tensor.new({1, 2, 3, 2, 1}, {5})
    local mask = t:eq(2)
    assert_eq(mask:get(0), 0, "1 == 2")
    assert_eq(mask:get(1), 1, "2 == 2")
    assert_eq(mask:get(3), 1, "2 == 2")
end)

-- ========================================
-- Level 8: 高级操作
-- ========================================
print("\n========== Level 8: 高级操作 ==========\n")

test("topk_new", function()
    local t = nn.Tensor.new({3, 1, 4, 1, 5, 9, 2, 6}, {8})
    local topk = t:topk_new(3, -1, true)
    assert_eq(topk.values[1], 9, "top1 value")
    assert_eq(topk.values[2], 6, "top2 value")
    assert_eq(topk.values[3], 5, "top3 value")
    assert_eq(topk.indices[1], 5, "top1 index")
end)

test("topk_new - smallest", function()
    local t = nn.Tensor.new({3, 1, 4, 1, 5, 9, 2, 6}, {8})
    local topk = t:topk_new(2, -1, false)
    assert_eq(topk.values[1], 1, "smallest1")
    assert_eq(topk.values[2], 1, "smallest2")
end)

test("nonzero", function()
    local t = nn.Tensor.new({0, 1, 0, 2, 0, 3}, {6})
    local indices = t:nonzero()
    assert_eq(#indices, 3, "nonzero count")
    assert_eq(indices[1], 1, "nonzero idx1")
    assert_eq(indices[2], 3, "nonzero idx2")
    assert_eq(indices[3], 5, "nonzero idx3")
end)

test("where_indices", function()
    local t = nn.Tensor.new({1, 5, 2, 8, 3, 9}, {6})
    local indices = t:where_indices(4.0, "gt")  -- > 4
    assert_eq(#indices, 3, "where count")  -- 5, 8, 9
end)

test("index_select", function()
    local t = nn.Tensor.new({10, 20, 30, 40, 50}, {5})
    local indices = {0, 2, 4}
    local selected = t:index_select(0, indices)
    assert_eq(selected:size(), 3, "selected size")
    assert_eq(selected:get(0), 10, "selected[0]")
    assert_eq(selected:get(1), 30, "selected[1]")
    assert_eq(selected:get(2), 50, "selected[2]")
end)

test("extract_columns", function()
    local t = nn.Tensor.new({1,2,3,4,5,6,7,8,9,10,11,12}, {3, 4})
    local cols = {0, 2}
    local extracted = t:extract_columns(cols)
    local shape = extracted:shape()
    assert_eq(shape[1], 3, "extracted rows")
    assert_eq(shape[2], 2, "extracted cols")
end)

-- ========================================
-- Level 9: 链式操作
-- ========================================
print("\n========== Level 9: 链式操作 ==========\n")

test("链式操作 - reshape后slice", function()
    local t = nn.Tensor.new({1,2,3,4,5,6,7,8,9,10,11,12}, {12})
    local result = t:reshape({3, 4}):slice(0, 0, 2, 1)
    local shape = result:shape()
    assert_eq(shape[1], 2, "chain shape[1]")
    assert_eq(shape[2], 4, "chain shape[2]")
end)

test("链式操作 - add后mul", function()
    local t = nn.Tensor.new({1, 2, 3}, {3})
    local result = t:add(1):mul(2)
    assert_eq(result:get(0), 4, "chain (1+1)*2")
    assert_eq(result:get(2), 8, "chain (3+1)*2")
end)

test("链式操作 - sigmoid后gt", function()
    local t = nn.Tensor.new({-2, 0, 2}, {3})
    local result = t:sigmoid():gt(0.5)
    assert_eq(result:get(0), 0, "sigmoid(-2) > 0.5")
    assert_eq(result:get(1), 0, "sigmoid(0) > 0.5")  -- exactly 0.5, not >
    assert_eq(result:get(2), 1, "sigmoid(2) > 0.5")
end)

-- ========================================
-- Level 10: 边界条件
-- ========================================
print("\n========== Level 10: 边界条件 ==========\n")

test("单元素tensor", function()
    local t = nn.Tensor.new({42}, {1})
    assert_eq(t:size(), 1, "single element size")
    assert_eq(t:get(0), 42, "single element value")
    local sum = t:sum(-1, false)
    assert_eq(sum:get(0), 42, "single element sum")
end)

test("大tensor性能", function()
    local data = {}
    for i = 1, 10000 do
        data[i] = i * 0.1
    end
    local t = nn.Tensor.new(data, {100, 100})
    local result = t:sum(-1, false)
    assert(result:get(0) > 0, "large tensor sum")
end)

test("负数处理", function()
    local t = nn.Tensor.new({-5, -3, -1, 0, 1, 3, 5}, {7})
    local max_val = t:max(-1, false)
    local min_val = t:min(-1, false)
    assert_eq(max_val:get(0), 5, "max with negatives")
    assert_eq(min_val:get(0), -5, "min with negatives")
end)

test("浮点数精度", function()
    local t = nn.Tensor.new({0.1, 0.2, 0.3}, {3})
    local sum = t:sum(-1, false)
    assert_near(sum:get(0), 0.6, 1e-5, "float sum precision")
end)

-- ========================================
-- Level 11: 非连续Tensor操作 (YOLOv5场景)
-- ========================================
print("\n========== Level 11: 非连续Tensor操作 ==========\n")

test("非连续Tensor - get_column", function()
    -- 模拟YOLOv5输出 [4, 5]: cx, cy, w, h, objectness
    local yolo_data = {
        10.0, 20.0, 5.0, 5.0, 0.8,   -- box 0
        15.0, 25.0, 6.0, 6.0, 0.3,   -- box 1
        20.0, 30.0, 7.0, 7.0, 0.9,   -- box 2
        25.0, 35.0, 8.0, 8.0, 0.2,   -- box 3
    }
    local yolo_t = nn.Tensor.new(yolo_data, {4, 5})
    local obj_col = yolo_t:get_column(4)
    local obj_table = obj_col:to_table()
    assert_near(obj_table[1], 0.8, 1e-5, "obj[0]")
    assert_near(obj_table[2], 0.3, 1e-5, "obj[1]")
    assert_near(obj_table[3], 0.9, 1e-5, "obj[2]")
    assert_near(obj_table[4], 0.2, 1e-5, "obj[3]")
end)

test("非连续Tensor - slice_columns", function()
    local yolo_data = {
        10.0, 20.0, 5.0, 5.0, 0.8,
        15.0, 25.0, 6.0, 6.0, 0.3,
        20.0, 30.0, 7.0, 7.0, 0.9,
        25.0, 35.0, 8.0, 8.0, 0.2,
    }
    local yolo_t = nn.Tensor.new(yolo_data, {4, 5})
    local boxes_col = yolo_t:slice_columns(0, 4)
    local shape = boxes_col:shape()
    assert_eq(shape[1], 4, "boxes rows")
    assert_eq(shape[2], 4, "boxes cols")
    local first_box = boxes_col:select_dim(0, 0):to_table()
    assert_near(first_box[1], 10.0, 1e-5, "box[0] cx")
    assert_near(first_box[2], 20.0, 1e-5, "box[0] cy")
    assert_near(first_box[3], 5.0, 1e-5, "box[0] w")
    assert_near(first_box[4], 5.0, 1e-5, "box[0] h")
end)

test("非连续Tensor - max操作", function()
    local yolo_data = {
        10.0, 20.0, 5.0, 5.0, 0.8,
        15.0, 25.0, 6.0, 6.0, 0.3,
        20.0, 30.0, 7.0, 7.0, 0.9,
        25.0, 35.0, 8.0, 8.0, 0.2,
    }
    local yolo_t = nn.Tensor.new(yolo_data, {4, 5})
    local obj_col = yolo_t:get_column(4)
    local max_obj = obj_col:max(-1, false)
    assert_near(max_obj:get(0), 0.9, 1e-5, "max objectness")
end)

test("非连续Tensor - slice_columns + max组合", function()
    local yolo_data = {
        10.0, 20.0, 5.0, 5.0, 0.8,
        15.0, 25.0, 6.0, 6.0, 0.3,
        20.0, 30.0, 7.0, 7.0, 0.9,
        25.0, 35.0, 8.0, 8.0, 0.2,
    }
    local yolo_t = nn.Tensor.new(yolo_data, {4, 5})
    -- 提取objectness列 (column 4)
    local obj_col = yolo_t:slice_columns(4, 5)
    local shape = obj_col:shape()
    assert_eq(shape[1], 4, "obj_col rows")
    assert_eq(shape[2], 1, "obj_col cols")
    -- squeeze to 1D and check values
    local obj_1d = obj_col:squeeze(-1)
    local expected = {0.8, 0.3, 0.9, 0.2}
    local obj_table = obj_1d:to_table()
    for i = 1, 4 do
        assert_near(obj_table[i], expected[i], 1e-5, "obj[" .. i .. "]")
    end
end)

test("非连续Tensor - slice_columns后沿axis求max (CRITICAL)", function()
    -- 这个测试覆盖YOLOv5的实际场景：slice_columns创建非连续view，然后沿axis求max
    local data = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
    }
    local t = nn.Tensor.new(data, {3, 5})

    -- 提取中间3列 [3, 3], 非连续, stride=[5,1]
    local cols = t:slice_columns(1, 4)
    assert_eq(cols:is_contiguous(), false, "slice_columns should be non-contiguous")

    -- 沿axis=1求max (每行的最大值)
    local max_vals = cols:max(1, false)  -- [3]

    -- 验证: 每行取最大值
    -- row 0: [2, 3, 4] -> max = 4
    -- row 1: [7, 8, 9] -> max = 9
    -- row 2: [12, 13, 14] -> max = 14
    assert_eq(max_vals:get(0), 4, "row 0 max")
    assert_eq(max_vals:get(1), 9, "row 1 max")
    assert_eq(max_vals:get(2), 14, "row 2 max")
end)

test("非连续Tensor - slice_columns后沿axis求argmax", function()
    local data = {
        1, 2, 5, 4, 3,    -- max at col 2 (value 5)
        6, 9, 8, 7, 10,   -- max at col 4 (value 10)
        11, 12, 15, 14, 13, -- max at col 2 (value 15)
    }
    local t = nn.Tensor.new(data, {3, 5})
    local cols = t:slice_columns(0, 5)  -- 全部列

    -- 沿axis=1求argmax
    local max_indices = cols:argmax(1)

    -- 验证索引
    assert_eq(max_indices[1], 2, "row 0 argmax")  -- col 2
    assert_eq(max_indices[2], 4, "row 1 argmax")  -- col 4
    assert_eq(max_indices[3], 2, "row 2 argmax")  -- col 2
end)

test("非连续Tensor - slice_columns后求sum", function()
    local data = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
    }
    local t = nn.Tensor.new(data, {2, 5})
    local cols = t:slice_columns(1, 4)  -- cols 1,2,3

    -- 沿axis=1求sum
    local sum_vals = cols:sum(1, false)

    -- row 0: 2+3+4 = 9
    -- row 1: 7+8+9 = 24
    assert_eq(sum_vals:get(0), 9, "row 0 sum")
    assert_eq(sum_vals:get(1), 24, "row 1 sum")
end)

test("非连续Tensor - slice_columns后求mean", function()
    local data = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
    }
    local t = nn.Tensor.new(data, {2, 5})
    local cols = t:slice_columns(1, 4)  -- cols 1,2,3

    -- 沿axis=1求mean
    local mean_vals = cols:mean(1, false)

    -- row 0: (2+3+4)/3 = 3
    -- row 1: (7+8+9)/3 = 8
    assert_eq(mean_vals:get(0), 3, "row 0 mean")
    assert_eq(mean_vals:get(1), 8, "row 1 mean")
end)

test("非连续Tensor - YOLOv5大规模场景", function()
    -- 模拟YOLOv5真实场景: [25200, 85] 提取类别分数 [25200, 80]
    local num_boxes = 100  -- 简化版，用100个boxes测试
    local data = {}
    for i = 1, num_boxes do
        -- 4个box坐标 + 1个objectness + 80个类别
        for j = 1, 85 do
            if j <= 4 then
                data[(i-1)*85 + j] = math.random() * 100  -- box coords
            elseif j == 5 then
                data[(i-1)*85 + j] = math.random()  -- objectness
            else
                data[(i-1)*85 + j] = math.random()  -- class scores
            end
        end
    end

    local t = nn.Tensor.new(data, {num_boxes, 85})

    -- 提取类别分数 [100, 80]
    local class_scores = t:slice_columns(5, 85)
    assert_eq(class_scores:is_contiguous(), false, "class_scores non-contiguous")

    -- 沿axis=1求max
    local max_scores = class_scores:max(1, false)
    assert_eq(max_scores:size(), num_boxes, "max_scores size")

    -- 验证所有分数在合理范围
    for i = 0, num_boxes - 1 do
        local score = max_scores:get(i)
        assert(score >= 0 and score <= 1, "score in valid range")
    end
end)

-- ========================================
-- Level 12: Gather/Concat/Split
-- ========================================
print("\n========== Level 12: Gather/Concat/Split ==========\n")

test("gather - 1D索引", function()
    -- source: [0, 10, 20, 30, 40]
    local src = nn.Tensor.new({0, 10, 20, 30, 40}, {5})
    -- indices: [4, 0, 2]
    local idx = nn.Tensor.new({4, 0, 2}, {3})
    local result = src:gather(0, idx)
    local shape = result:shape()
    assert_eq(shape[1], 3, "gather 1D shape")
    assert_eq(result:get(0), 40, "gather[0]")
    assert_eq(result:get(1), 0, "gather[1]")
    assert_eq(result:get(2), 20, "gather[2]")
end)

test("gather - 2D沿axis=1", function()
    -- source: [[1,2,3], [4,5,6]]
    local src = nn.Tensor.new({1,2,3,4,5,6}, {2, 3})
    -- indices: [[0,2], [1,0]] - gather from each row
    local idx = nn.Tensor.new({0, 2, 1, 0}, {2, 2})
    local result = src:gather(1, idx)
    local shape = result:shape()
    assert_eq(shape[1], 2, "gather 2D rows")
    assert_eq(shape[2], 2, "gather 2D cols")
    assert_eq(result:get(0), 1, "gather[0,0] = src[0,0]")
    assert_eq(result:get(1), 3, "gather[0,1] = src[0,2]")
    assert_eq(result:get(2), 5, "gather[1,0] = src[1,1]")
    assert_eq(result:get(3), 4, "gather[1,1] = src[1,0]")
end)

test("concat - 沿axis=0", function()
    local t1 = nn.Tensor.new({1, 2, 3}, {1, 3})
    local t2 = nn.Tensor.new({4, 5, 6}, {1, 3})
    local t3 = nn.Tensor.new({7, 8, 9}, {1, 3})
    local result = nn.Tensor.concat({t1, t2, t3}, 0)
    local shape = result:shape()
    assert_eq(shape[1], 3, "concat axis=0 rows")
    assert_eq(shape[2], 3, "concat axis=0 cols")
    assert_eq(result:get(0), 1, "concat[0,0]")
    assert_eq(result:get(3), 4, "concat[1,0]")
    assert_eq(result:get(6), 7, "concat[2,0]")
end)

test("concat - 沿axis=1", function()
    local t1 = nn.Tensor.new({1, 2, 3, 4}, {2, 2})
    local t2 = nn.Tensor.new({5, 6, 7, 8}, {2, 2})
    local result = nn.Tensor.concat({t1, t2}, 1)
    local shape = result:shape()
    assert_eq(shape[1], 2, "concat axis=1 rows")
    assert_eq(shape[2], 4, "concat axis=1 cols")
    -- Row 0: [1, 2, 5, 6]
    assert_eq(result:get(0), 1, "concat[0,0]")
    assert_eq(result:get(1), 2, "concat[0,1]")
    assert_eq(result:get(2), 5, "concat[0,2]")
    assert_eq(result:get(3), 6, "concat[0,3]")
end)

test("split - 均匀分割", function()
    local t = nn.Tensor.new({1,2,3,4,5,6,7,8,9,10,11,12}, {4, 3})
    local parts = t:split(2, 0)  -- Split into 2 parts along axis 0
    assert_eq(#parts, 2, "split count")
    local shape1 = parts[1]:shape()
    local shape2 = parts[2]:shape()
    assert_eq(shape1[1], 2, "split[0] rows")
    assert_eq(shape1[2], 3, "split[0] cols")
    assert_eq(shape2[1], 2, "split[1] rows")
    -- First part: rows 0-1
    assert_eq(parts[1]:get(0), 1, "split[0][0,0]")
    assert_eq(parts[1]:get(3), 4, "split[0][1,0]")
    -- Second part: rows 2-3
    assert_eq(parts[2]:get(0), 7, "split[1][0,0]")
    assert_eq(parts[2]:get(3), 10, "split[1][1,0]")
end)

test("split - 沿axis=1", function()
    local t = nn.Tensor.new({1,2,3,4,5,6,7,8}, {2, 4})
    local parts = t:split(2, 1)  -- Split into 2 parts along axis 1
    assert_eq(#parts, 2, "split count")
    local shape = parts[1]:shape()
    assert_eq(shape[1], 2, "split rows")
    assert_eq(shape[2], 2, "split cols")
    -- First part: cols 0-1 -> [[1,2], [5,6]]
    assert_eq(parts[1]:get(0), 1, "split[0][0,0]")
    assert_eq(parts[1]:get(1), 2, "split[0][0,1]")
    -- Second part: cols 2-3 -> [[3,4], [7,8]]
    assert_eq(parts[2]:get(0), 3, "split[1][0,0]")
    assert_eq(parts[2]:get(1), 4, "split[1][0,1]")
end)

test("concat + split 往返", function()
    local t1 = nn.Tensor.new({1, 2, 3, 4}, {2, 2})
    local t2 = nn.Tensor.new({5, 6, 7, 8}, {2, 2})
    -- Concat then split should give back original
    local concat_result = nn.Tensor.concat({t1, t2}, 0)
    local parts = concat_result:split(2, 0)
    assert_eq(parts[1]:get(0), 1, "roundtrip[0][0]")
    assert_eq(parts[1]:get(3), 4, "roundtrip[0][3]")
    assert_eq(parts[2]:get(0), 5, "roundtrip[1][0]")
    assert_eq(parts[2]:get(3), 8, "roundtrip[1][3]")
end)

-- 总结
print("\n========================================")
print(string.format("测试总结: %d/%d 通过", pass_count, test_count))
print("========================================\n")

if pass_count == test_count then
    print("✓ 所有测试通过!")
else
    print("✗ 有测试失败，请检查上面的错误信息")
end

return pass_count == test_count
