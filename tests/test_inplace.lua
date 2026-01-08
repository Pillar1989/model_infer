-- Level 3.5: In-place 操作
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq

local nn = lua_nn

print("\n========== Level 3.5: In-place 操作 ==========\n")

test("add_ in-place", function()
    local t = nn.Tensor.new({1, 2, 3}, {3})
    t:add_(10)
    assert_eq(t:get(0), 11, "add_ result[0]")
    assert_eq(t:get(2), 13, "add_ result[2]")
end)

test("sub_ in-place", function()
    local t = nn.Tensor.new({10, 20, 30}, {3})
    t:sub_(5)
    assert_eq(t:get(0), 5, "sub_ result[0]")
    assert_eq(t:get(2), 25, "sub_ result[2]")
end)

test("mul_ in-place", function()
    local t = nn.Tensor.new({2, 3, 4}, {3})
    t:mul_(10)
    assert_eq(t:get(0), 20, "mul_ result[0]")
    assert_eq(t:get(2), 40, "mul_ result[2]")
end)

test("div_ in-place", function()
    local t = nn.Tensor.new({10, 20, 30}, {3})
    t:div_(10)
    assert_eq(t:get(0), 1, "div_ result[0]")
    assert_eq(t:get(2), 3, "div_ result[2]")
end)

return true
