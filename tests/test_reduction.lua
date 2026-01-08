-- Level 5: 归约操作
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq
local assert_near = helpers.assert_near

local nn = lua_nn

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
    assert_eq(max_idx, 5, "argmax index")
end)

test("argmin", function()
    local t = nn.Tensor.new({3, 1, 4, 1, 5, 9, 2, 6}, {8})
    local min_idx = t:argmin(-1)
    assert_eq(min_idx, 1, "argmin index")
end)

test("max_with_argmax - axis 0", function()
    local t = nn.Tensor.new({1,4,7,10, 2,5,8,11, 3,6,9,12}, {3, 4})
    local result = t:max_with_argmax(0)

    local values = result.values
    local indices = result.indices

    assert_eq(values:size(), 4, "values size")
    assert_eq(#indices, 4, "indices size")

    assert_near(values:get(0), 3, 1e-5, "max[0]")
    assert_near(values:get(1), 6, 1e-5, "max[1]")
    assert_near(values:get(2), 9, 1e-5, "max[2]")
    assert_near(values:get(3), 12, 1e-5, "max[3]")

    assert_eq(indices[1], 2, "argmax[0]")
    assert_eq(indices[2], 2, "argmax[1]")
    assert_eq(indices[3], 2, "argmax[2]")
    assert_eq(indices[4], 2, "argmax[3]")
end)

test("max_with_argmax - axis 1", function()
    local t = nn.Tensor.new({1,4,7,10, 2,5,8,11, 3,6,9,12}, {3, 4})
    local result = t:max_with_argmax(1)

    local values = result.values
    local indices = result.indices

    assert_eq(values:size(), 3, "values size")
    assert_eq(#indices, 3, "indices size")

    assert_near(values:get(0), 10, 1e-5, "max[0]")
    assert_near(values:get(1), 11, 1e-5, "max[1]")
    assert_near(values:get(2), 12, 1e-5, "max[2]")

    assert_eq(indices[1], 3, "argmax[0]")
    assert_eq(indices[2], 3, "argmax[1]")
    assert_eq(indices[3], 3, "argmax[2]")
end)

return true
