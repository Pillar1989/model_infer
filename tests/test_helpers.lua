-- Test Helper Functions
-- 共享的测试工具函数

local test_count = 0
local pass_count = 0

local function test(name, fn)
    test_count = test_count + 1
    io.write(string.format("测试%d: %s ... ", test_count, name))
    local ok, err = pcall(fn)
    if ok then
        pass_count = pass_count + 1
        print("✓")
    else
        print("✗")
        print("  错误:", err)
    end
end

local function assert_eq(a, b, msg)
    if a ~= b then
        error(string.format("%s: expected %s, got %s", msg or "assertion failed", tostring(b), tostring(a)))
    end
end

local function assert_near(a, b, eps, msg)
    eps = eps or 1e-5
    if math.abs(a - b) > eps then
        error(string.format("%s: expected ~%s, got %s", msg or "assertion failed", tostring(b), tostring(a)))
    end
end

local function get_stats()
    return test_count, pass_count
end

local function reset_stats()
    test_count = 0
    pass_count = 0
end

return {
    test = test,
    assert_eq = assert_eq,
    assert_near = assert_near,
    get_stats = get_stats,
    reset_stats = reset_stats
}
