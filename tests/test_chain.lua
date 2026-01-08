-- Level 9: 链式操作
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq

local nn = lua_nn

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
    assert_eq(result:get(1), 0, "sigmoid(0) > 0.5")
    assert_eq(result:get(2), 1, "sigmoid(2) > 0.5")
end)

return true
