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

-- Level 1: 基础形状操作
print("\n========== Level 1: 基础形状操作 ==========\n")

test("基础构造", function()
    local t = nn.Tensor({1, 2, 3, 4, 5, 6}, {2, 3})
    assert(t.ndim == 2)
    assert(t:size() == 6)
end)

test("Slice操作", function()
    local t = nn.Tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, {3, 4})
    local sliced = t:slice(0, 0, 2, 1)  -- 添加step参数
    local shape = sliced:shape()
    assert(shape[1] == 2 and shape[2] == 4)
end)

test("Reshape操作", function()
    local t = nn.Tensor({1, 2, 3, 4, 5, 6}, {2, 3})
    local reshaped = t:reshape({3, 2})
    local shape = reshaped:shape()
    assert(shape[1] == 3 and shape[2] == 2)
end)

test("Transpose操作", function()
    local t = nn.Tensor({1, 2, 3, 4, 5, 6}, {2, 3})
    local transposed = t:transpose()
    local shape = transposed:shape()
    assert(shape[1] == 3 and shape[2] == 2)
end)

-- Level 2: 数学运算
print("\n========== Level 2: 数学运算 ==========\n")

test("Element-wise加法", function()
    local t = nn.Tensor({1, 2, 3, 4}, {4})
    local result = t:add(10)
    print("    结果: " .. tostring(result))
end)

test("Sigmoid", function()
    local t = nn.Tensor({-1, 0, 1, 2}, {4})
    local result = t:sigmoid()
    print("    结果: " .. tostring(result))
end)

test("Softmax", function()
    local t = nn.Tensor({-1, 0, 1, 2}, {4})
    local result = t:softmax(-1)
    print("    结果: " .. tostring(result))
end)

test("Argmax", function()
    local t = nn.Tensor({3, 1, 4, 1, 5, 9, 2, 6}, {8})
    local max_idx = t:argmax(-1)
    assert(max_idx == 5)
end)

-- Level 3: 高级操作
print("\n========== Level 3: 高级操作 ==========\n")

test("TopK", function()
    local t = nn.Tensor({3, 1, 4, 1, 5, 9, 2, 6}, {8})
    local topk = t:topk_new(3, -1, true)
    assert(topk.values[1] == 9)
    print("    Top 3: " .. topk.values[1] .. ", " .. topk.values[2] .. ", " .. topk.values[3])
end)

test("to_table", function()
    local t = nn.Tensor({1, 2, 3, 4}, {4})
    local tbl = t:to_table()
    assert(#tbl == 4)
end)

-- 总结
print("\n========================================")
print(string.format("测试总结: %d/%d 通过", pass_count, test_count))
print("========================================\n")

return pass_count == test_count
