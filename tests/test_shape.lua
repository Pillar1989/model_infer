-- Level 2: 形状操作
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq

local nn = lua_nn

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

return true
