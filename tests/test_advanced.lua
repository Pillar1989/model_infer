-- Level 8: 高级操作
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq

local nn = lua_nn

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
    local indices = t:where_indices(4.0, "gt")
    assert_eq(#indices, 3, "where count")
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
    assert_eq(#extracted, 2, "extracted columns count")
    assert_eq(#extracted[1], 3, "column 0 rows")
    assert_eq(extracted[1][1], 1, "col0 row0")
    assert_eq(extracted[1][2], 5, "col0 row1")
    assert_eq(extracted[1][3], 9, "col0 row2")
    assert_eq(extracted[2][1], 3, "col2 row0")
    assert_eq(extracted[2][2], 7, "col2 row1")
    assert_eq(extracted[2][3], 11, "col2 row2")
end)

return true
