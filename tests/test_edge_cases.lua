-- Level 10: 边界条件
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq
local assert_near = helpers.assert_near

local nn = lua_nn

print("\n========== Level 10: 边界条件 ==========\n")

test("单元素tensor", function()
    local t = nn.Tensor.new({42}, {1})
    assert_eq(t:size(), 1, "single element size")
    assert_eq(t:get(0), 42, "single element value")
    local sum = t:sum(-1, false)
    assert_eq(sum:get(0), 42, "single element sum")
end)

test("大tensor性能", function()
    local data = {}
    for i = 1, 10000 do
        data[i] = i * 0.1
    end
    local t = nn.Tensor.new(data, {100, 100})
    local result = t:sum(-1, false)
    assert(result:get(0) > 0, "large tensor sum")
end)

test("负数处理", function()
    local t = nn.Tensor.new({-5, -3, -1, 0, 1, 3, 5}, {7})
    local max_val = t:max(-1, false)
    local min_val = t:min(-1, false)
    assert_eq(max_val:get(0), 5, "max with negatives")
    assert_eq(min_val:get(0), -5, "min with negatives")
end)

test("浮点数精度", function()
    local t = nn.Tensor.new({0.1, 0.2, 0.3}, {3})
    local sum = t:sum(-1, false)
    assert_near(sum:get(0), 0.6, 1e-5, "float sum precision")
end)

return true
