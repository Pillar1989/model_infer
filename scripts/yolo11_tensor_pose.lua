-- YOLO11 Pose Estimation Script (使用新Tensor API)
-- 展示如何用通用Tensor操作处理姿态估计
local utils = lua_utils
local nn = lua_nn

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
    num_keypoints = 17,  -- COCO格式17个关键点
    keypoint_names = {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    },
    skeleton = {
        {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13},
        {6, 12}, {7, 13}, {6, 7}, {6, 8}, {7, 9},
        {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3},
        {2, 4}, {3, 5}, {4, 6}, {5, 7}
    }
}

function Model.preprocess(img)
    local w, h = img.width, img.height
    local target_h, target_w = table.unpack(Model.config.input_size)

    local r = math.min(target_h / h, target_w / w)
    local new_w = math.floor(w * r)
    local new_h = math.floor(h * r)

    if new_w ~= w or new_h ~= h then
        img:resize(new_w, new_h)
    end

    local dw = target_w - new_w
    local dh = target_h - new_h
    
    dw = dw % Model.config.stride
    dh = dh % Model.config.stride

    local top = math.floor(dh / 2)
    local bottom = dh - top
    local left = math.floor(dw / 2)
    local right = dw - left

    img:pad(top, bottom, left, right, 114)

    local scale = 1.0 / 255.0
    local input_tensor = img:to_tensor(scale, {0,0,0}, {1,1,1})

    local meta = {
        scale = r,
        pad_x = left,
        pad_y = top,
        ori_w = w,
        ori_h = h
    }

    return input_tensor, meta
end

-- 使用向量化Tensor操作实现姿态估计后处理（方案3 - 极致性能）
function Model.postprocess(outputs, meta)
    local output = outputs["output0"]

    if not output then
        error("Missing output0")
    end

    -- YOLO11-Pose格式: [1, 56, 8400]
    -- 4个box坐标 + 1个person类别分数 + 51个关键点数据(17个点*3: x,y,conf)
    local num_kpt = Model.config.num_keypoints

    -- 1. 分离boxes, scores, keypoints
    local boxes = output:slice(1, 0, 4, 1):squeeze(0)  -- [4, 8400]
    local person_scores = output:slice(1, 4, 5, 1):squeeze(0):squeeze(0)  -- [8400]
    local keypoints = output:slice(1, 5, 56, 1):squeeze(0)  -- [51, 8400]

    -- 2. 向量化过滤：找出满足条件的索引
    local valid_indices = person_scores:where_indices(Model.config.conf_thres, "ge")

    print(string.format("过滤后候选框: %d", #valid_indices))

    if #valid_indices == 0 then
        return {}
    end

    -- 3. 批量提取有效数据
    local filtered_boxes = boxes:extract_columns(valid_indices)  -- {{cx,cy,w,h}, ...}
    local filtered_scores_tensor = person_scores:index_select(0, valid_indices)
    local filtered_scores = filtered_scores_tensor:to_table()
    local filtered_keypoints = keypoints:extract_columns(valid_indices)  -- {{kpt_data...}, ...}

    local proposals = {}

    -- 4. 只遍历过滤后的小数据集
    for i = 1, #valid_indices do
        local box_data = filtered_boxes[i]
        local kpt_data = filtered_keypoints[i]

        local cx = box_data[1]
        local cy = box_data[2]
        local w = box_data[3]
        local h = box_data[4]

        -- 将中心点坐标转换为左上角坐标
        local x = cx - w / 2.0
        local y = cy - h / 2.0

        -- 提取17个关键点(每个3个值: x, y, conf)
        local kpts = {}
        for j = 1, num_kpt do
            local idx_base = (j - 1) * 3
            local kpt_x = kpt_data[idx_base + 1] or 0
            local kpt_y = kpt_data[idx_base + 2] or 0
            local kpt_c = kpt_data[idx_base + 3] or 0

            kpts[j] = {
                x = kpt_x,
                y = kpt_y,
                v = kpt_c,
                conf = kpt_c,
                name = Model.config.keypoint_names[j]
            }
        end

        table.insert(proposals, {
            x = x,
            y = y,
            w = w,
            h = h,
            score = filtered_scores[i],
            class_id = 0,  -- person类别
            label = "person",
            keypoints = kpts
        })
    end
    
    -- 3. 坐标还原(包括关键点)
    for _, box in ipairs(proposals) do
        box.x = (box.x - meta.pad_x) / meta.scale
        box.y = (box.y - meta.pad_y) / meta.scale
        box.w = box.w / meta.scale
        box.h = box.h / meta.scale
        
        -- 还原关键点坐标
        for _, kpt in ipairs(box.keypoints) do
            kpt.x = (kpt.x - meta.pad_x) / meta.scale
            kpt.y = (kpt.y - meta.pad_y) / meta.scale
        end
    end
    
    -- 4. NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    
    print(string.format("NMS后最终框: %d", #final_boxes))
    
    return final_boxes
end

return Model
