-- Level 4: 数学运算 - Tensor
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq

local nn = lua_nn

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

return true
