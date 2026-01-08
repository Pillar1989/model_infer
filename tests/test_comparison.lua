-- Level 7: 比较操作
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq

local nn = lua_nn

print("\n========== Level 7: 比较操作 ==========\n")

test("gt (>)", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5}, {5})
    local mask = t:gt(3)
    assert_eq(mask:get(0), 0, "1 > 3")
    assert_eq(mask:get(2), 0, "3 > 3")
    assert_eq(mask:get(3), 1, "4 > 3")
    assert_eq(mask:get(4), 1, "5 > 3")
end)

test("lt (<)", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5}, {5})
    local mask = t:lt(3)
    assert_eq(mask:get(0), 1, "1 < 3")
    assert_eq(mask:get(1), 1, "2 < 3")
    assert_eq(mask:get(2), 0, "3 < 3")
end)

test("ge (>=)", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5}, {5})
    local mask = t:ge(3)
    assert_eq(mask:get(1), 0, "2 >= 3")
    assert_eq(mask:get(2), 1, "3 >= 3")
    assert_eq(mask:get(3), 1, "4 >= 3")
end)

test("le (<=)", function()
    local t = nn.Tensor.new({1, 2, 3, 4, 5}, {5})
    local mask = t:le(3)
    assert_eq(mask:get(0), 1, "1 <= 3")
    assert_eq(mask:get(2), 1, "3 <= 3")
    assert_eq(mask:get(3), 0, "4 <= 3")
end)

test("eq (==)", function()
    local t = nn.Tensor.new({1, 2, 3, 2, 1}, {5})
    local mask = t:eq(2)
    assert_eq(mask:get(0), 0, "1 == 2")
    assert_eq(mask:get(1), 1, "2 == 2")
    assert_eq(mask:get(3), 1, "2 == 2")
end)

return true
