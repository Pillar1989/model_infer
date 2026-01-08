-- Level 1: 基础属性和构造
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq

local nn = lua_nn

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

return true
