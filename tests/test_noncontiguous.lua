-- Level 11: 非连续Tensor操作 (YOLOv5场景)
local helpers = require("tests.test_helpers")
local test = helpers.test
local assert_eq = helpers.assert_eq
local assert_near = helpers.assert_near

local nn = lua_nn

print("\n========== Level 11: 非连续Tensor操作 ==========\n")

test("非连续Tensor - get_column", function()
    local yolo_data = {
        10.0, 20.0, 5.0, 5.0, 0.8,
        15.0, 25.0, 6.0, 6.0, 0.3,
        20.0, 30.0, 7.0, 7.0, 0.9,
        25.0, 35.0, 8.0, 8.0, 0.2,
    }
    local yolo_t = nn.Tensor.new(yolo_data, {4, 5})
    local obj_col = yolo_t:get_column(4)
    local obj_table = obj_col:to_table()
    assert_near(obj_table[1], 0.8, 1e-5, "obj[0]")
    assert_near(obj_table[2], 0.3, 1e-5, "obj[1]")
    assert_near(obj_table[3], 0.9, 1e-5, "obj[2]")
    assert_near(obj_table[4], 0.2, 1e-5, "obj[3]")
end)

test("非连续Tensor - slice_columns", function()
    local yolo_data = {
        10.0, 20.0, 5.0, 5.0, 0.8,
        15.0, 25.0, 6.0, 6.0, 0.3,
        20.0, 30.0, 7.0, 7.0, 0.9,
        25.0, 35.0, 8.0, 8.0, 0.2,
    }
    local yolo_t = nn.Tensor.new(yolo_data, {4, 5})
    local boxes_col = yolo_t:slice_columns(0, 4)
    local shape = boxes_col:shape()
    assert_eq(shape[1], 4, "boxes rows")
    assert_eq(shape[2], 4, "boxes cols")
    local first_box = boxes_col:select_dim(0, 0):to_table()
    assert_near(first_box[1], 10.0, 1e-5, "box[0] cx")
    assert_near(first_box[2], 20.0, 1e-5, "box[0] cy")
    assert_near(first_box[3], 5.0, 1e-5, "box[0] w")
    assert_near(first_box[4], 5.0, 1e-5, "box[0] h")
end)

test("非连续Tensor - max操作", function()
    local yolo_data = {
        10.0, 20.0, 5.0, 5.0, 0.8,
        15.0, 25.0, 6.0, 6.0, 0.3,
        20.0, 30.0, 7.0, 7.0, 0.9,
        25.0, 35.0, 8.0, 8.0, 0.2,
    }
    local yolo_t = nn.Tensor.new(yolo_data, {4, 5})
    local obj_col = yolo_t:get_column(4)
    local max_obj = obj_col:max(-1, false)
    assert_near(max_obj:get(0), 0.9, 1e-5, "max objectness")
end)

test("非连续Tensor - slice_columns + max组合", function()
    local yolo_data = {
        10.0, 20.0, 5.0, 5.0, 0.8,
        15.0, 25.0, 6.0, 6.0, 0.3,
        20.0, 30.0, 7.0, 7.0, 0.9,
        25.0, 35.0, 8.0, 8.0, 0.2,
    }
    local yolo_t = nn.Tensor.new(yolo_data, {4, 5})
    local obj_col = yolo_t:slice_columns(4, 5)
    local shape = obj_col:shape()
    assert_eq(shape[1], 4, "obj_col rows")
    assert_eq(shape[2], 1, "obj_col cols")
    local obj_1d = obj_col:squeeze(-1)
    local expected = {0.8, 0.3, 0.9, 0.2}
    local obj_table = obj_1d:to_table()
    for i = 1, 4 do
        assert_near(obj_table[i], expected[i], 1e-5, "obj[" .. i .. "]")
    end
end)

test("非连续Tensor - slice_columns后沿axis求max (CRITICAL)", function()
    local data = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
    }
    local t = nn.Tensor.new(data, {3, 5})

    local cols = t:slice_columns(1, 4)
    assert_eq(cols:is_contiguous(), false, "slice_columns should be non-contiguous")

    local max_vals = cols:max(1, false)

    assert_eq(max_vals:get(0), 4, "row 0 max")
    assert_eq(max_vals:get(1), 9, "row 1 max")
    assert_eq(max_vals:get(2), 14, "row 2 max")
end)

test("非连续Tensor - slice_columns后沿axis求argmax", function()
    local data = {
        1, 2, 5, 4, 3,
        6, 9, 8, 7, 10,
        11, 12, 15, 14, 13,
    }
    local t = nn.Tensor.new(data, {3, 5})
    local cols = t:slice_columns(0, 5)

    local max_indices = cols:argmax(1)

    assert_eq(max_indices[1], 2, "row 0 argmax")
    assert_eq(max_indices[2], 4, "row 1 argmax")
    assert_eq(max_indices[3], 2, "row 2 argmax")
end)

test("非连续Tensor - slice_columns后求sum", function()
    local data = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
    }
    local t = nn.Tensor.new(data, {2, 5})
    local cols = t:slice_columns(1, 4)

    local sum_vals = cols:sum(1, false)

    assert_eq(sum_vals:get(0), 9, "row 0 sum")
    assert_eq(sum_vals:get(1), 24, "row 1 sum")
end)

test("非连续Tensor - slice_columns后求mean", function()
    local data = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
    }
    local t = nn.Tensor.new(data, {2, 5})
    local cols = t:slice_columns(1, 4)

    local mean_vals = cols:mean(1, false)

    assert_eq(mean_vals:get(0), 3, "row 0 mean")
    assert_eq(mean_vals:get(1), 8, "row 1 mean")
end)

test("非连续Tensor - YOLOv5大规模场景", function()
    local num_boxes = 100
    local data = {}
    for i = 1, num_boxes do
        for j = 1, 85 do
            if j <= 4 then
                data[(i-1)*85 + j] = math.random() * 100
            elseif j == 5 then
                data[(i-1)*85 + j] = math.random()
            else
                data[(i-1)*85 + j] = math.random()
            end
        end
    end

    local t = nn.Tensor.new(data, {num_boxes, 85})

    local class_scores = t:slice_columns(5, 85)
    assert_eq(class_scores:is_contiguous(), false, "class_scores non-contiguous")

    local max_scores = class_scores:max(1, false)
    assert_eq(max_scores:size(), num_boxes, "max_scores size")

    for i = 0, num_boxes - 1 do
        local score = max_scores:get(i)
        assert(score >= 0 and score <= 1, "score in valid range")
    end
end)

return true
