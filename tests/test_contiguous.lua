-- Level 2.5: Contiguous标记验证
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq

local nn = lua_nn

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
    assert_eq(cont:get(0), 1, "values preserved")
end)

test("contiguous方法 - 非连续", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local cols = t:slice_columns(1, 3)  -- 非连续
    assert_eq(cols:is_contiguous(), false, "before contiguous()")

    local cont = cols:contiguous()
    assert(cont:is_contiguous(), "after contiguous() should be contiguous")

    local row0 = cont:select_dim(0, 0):to_table()
    assert_eq(row0[1], 2, "row 0 col 0")
    assert_eq(row0[2], 3, "row 0 col 1")
end)

test("view操作组合 - slice后slice", function()
    local t = nn.Tensor.new({1,2,3,4,5,6,7,8,9,10,11,12}, {3, 4})
    local sliced1 = t:slice(0, 0, 3, 1)  -- 全部行
    local sliced2 = sliced1:slice(1, 1, 3, 1)  -- 中间2列
    assert_eq(sliced2:is_contiguous(), false, "double slice non-contiguous")

    local val = sliced2:get(0)  -- row 0, col 1 (原始) = 2
    assert_eq(val, 2, "double slice value")
end)

test("view操作组合 - transpose后slice", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5, 6}, {2, 3})
    local transposed = t:transpose()  -- [3, 2]
    local sliced = transposed:slice(0, 0, 2, 1)  -- 前2行
    assert_eq(sliced:is_contiguous(), false, "transpose then slice")

    local shape = sliced:shape()
    assert_eq(shape[1], 2, "sliced shape[0]")
    assert_eq(shape[2], 2, "sliced shape[1]")
end)

return true
