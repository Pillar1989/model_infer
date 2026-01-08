-- Tensor API 测试运行器
-- 使用方式: ./build/test_tensor tests/run_all_tests.lua

local helpers = require("tests.test_helpers")

print("========================================")
print("    Tensor API 完整测试")
print("========================================\n")

-- 所有测试模块列表
local test_modules = {
    "tests.test_basic",
    "tests.test_shape",
    "tests.test_contiguous",
    "tests.test_scalar_ops",
    "tests.test_inplace",
    "tests.test_tensor_ops",
    "tests.test_reduction",
    "tests.test_activation",
    "tests.test_comparison",
    "tests.test_advanced",
    "tests.test_chain",
    "tests.test_edge_cases",
    "tests.test_noncontiguous",
    "tests.test_gather_concat",
}

-- 运行所有测试模块
for _, module_name in ipairs(test_modules) do
    local ok, result = pcall(require, module_name)
    if not ok then
        print("\n✗ 模块加载失败:", module_name)
        print("  错误:", result)
        os.exit(1)
    end
end

-- 获取最终统计
local test_count, pass_count = helpers.get_stats()

-- 总结
print("\n========================================")
print(string.format("测试总结: %d/%d 通过", pass_count, test_count))
print("========================================\n")

if pass_count == test_count then
    print("✓ 所有测试通过!")
else
    print("✗ 有测试失败，请检查上面的错误信息")
end

return pass_count == test_count
