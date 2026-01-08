-- Level 3: 数学运算 - 标量
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq

local nn = lua_nn

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

return true
