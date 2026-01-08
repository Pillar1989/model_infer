-- Level 6: 激活函数
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_near = helpers.assert_near

local nn = lua_nn

print("\n========== Level 6: 激活函数 ==========\n")

test("sigmoid", function()
    local t = nn.Tensor.new({0}, {1})
    local result = t:sigmoid()
    assert_near(result:get(0), 0.5, 1e-5, "sigmoid(0)")
end)

test("sigmoid - 多值", function()
    local t = nn.Tensor.new({-1, 0, 1, 2}, {4})
    local result = t:sigmoid()
    assert_near(result:get(1), 0.5, 1e-5, "sigmoid middle")
    assert(result:get(0) < 0.5, "sigmoid negative < 0.5")
    assert(result:get(2) > 0.5, "sigmoid positive > 0.5")
end)

test("softmax", function()
    local t = nn.Tensor.new({1, 2, 3}, {3})
    local result = t:softmax(-1)
    local sum = result:get(0) + result:get(1) + result:get(2)
    assert_near(sum, 1.0, 1e-5, "softmax sum")
end)

test("exp", function()
    local t = nn.Tensor.new({0, 1}, {2})
    local result = t:exp()
    assert_near(result:get(0), 1.0, 1e-5, "exp(0)")
    assert_near(result:get(1), 2.718281828, 1e-5, "exp(1)")
end)

test("log", function()
    local t = nn.Tensor.new({1, 2.718281828}, {2})
    local result = t:log()
    assert_near(result:get(0), 0.0, 1e-5, "log(1)")
    assert_near(result:get(1), 1.0, 1e-5, "log(e)")
end)

return true
