-- YOLO11 Instance Segmentation Script (使用新Tensor API)
-- 展示如何用通用Tensor操作处理分割任务
local utils = lua_utils
local nn = lua_nn

local Model = {}

Model.config = {
    input_size = {640, 640},
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
    labels = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
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
        ori_h = h,
        input_w = img.width,
        input_h = img.height
    }

    return input_tensor, meta
end

-- 使用向量化Tensor操作实现分割后处理（方案3 - 极致性能）
function Model.postprocess(outputs, meta)
    local output0 = outputs["output0"]  -- [1, 116, 8400]: boxes + 32 mask coeffs
    local output1 = outputs["output1"]  -- [1, 32, 160, 160]: proto masks

    if not output0 or not output1 then
        error("Missing outputs")
    end

    -- YOLO11-Seg格式: output0=[1, 116, 8400], output1=[1, 32, 160, 160]
    local num_mask_coeffs = 32
    local num_classes = 80

    -- 1. 分离boxes, scores, mask_coeffs
    local boxes = output0:slice(1, 0, 4, 1):squeeze(0)  -- [4, 8400]
    local scores = output0:slice(1, 4, 4 + num_classes, 1):squeeze(0)  -- [80, 8400]
    local mask_coeffs = output0:slice(1, 4 + num_classes, 116, 1):squeeze(0)  -- [32, 8400]

    -- 2. 找到最大分数和类别
    local max_scores = scores:max(0, false)  -- [8400]
    local class_ids = scores:argmax(0)  -- Lua table [8400]

    -- 3. 向量化过滤：找出满足条件的索引
    local valid_indices = max_scores:where_indices(Model.config.conf_thres, "ge")

    print(string.format("过滤后候选框: %d", #valid_indices))

    if #valid_indices == 0 then
        return {}
    end

    -- 4. 批量提取有效数据
    local filtered_boxes = boxes:extract_columns(valid_indices)  -- {{cx,cy,w,h}, ...}
    local filtered_scores_tensor = max_scores:index_select(0, valid_indices)
    local filtered_scores = filtered_scores_tensor:to_table()
    local filtered_mask_coeffs = mask_coeffs:extract_columns(valid_indices)  -- {{coeff1, ...coeff32}, ...}

    local proposals = {}

    -- 5. 只遍历过滤后的小数据集
    for i = 1, #valid_indices do
        local idx = valid_indices[i]
        local box_data = filtered_boxes[i]
        local coeffs = filtered_mask_coeffs[i]  -- 直接获取32个系数

        local cx = box_data[1]
        local cy = box_data[2]
        local w = box_data[3]
        local h = box_data[4]
        local cls_id = class_ids[idx + 1]  -- Lua索引从1开始

        -- 将中心点坐标转换为左上角坐标
        local x = cx - w / 2.0
        local y = cy - h / 2.0

        table.insert(proposals, {
            x = x,
            y = y,
            w = w,
            h = h,
            score = filtered_scores[i],
            class_id = cls_id,
            label = Model.config.labels[cls_id + 1] or "unknown",
            mask_coeffs = coeffs
        })
    end
    
    -- 4. 坐标还原
    for _, box in ipairs(proposals) do
        box.x = (box.x - meta.pad_x) / meta.scale
        box.y = (box.y - meta.pad_y) / meta.scale
        box.w = box.w / meta.scale
        box.h = box.h / meta.scale
    end
    
    -- 5. NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    
    print(string.format("NMS后最终框: %d", #final_boxes))
    
    -- 6. 生成mask (使用legacy方法，因为mask生成涉及复杂的矩阵乘法)
    -- TODO: 后续可以用tensor操作实现 mask = sigmoid(coeffs @ proto_masks)
    for i = 1, #final_boxes do
        local box = final_boxes[i]
        -- 暂时使用旧的process_mask方法
        local mask_tensor = output1:process_mask(
            box.mask_coeffs, box, 
            meta.ori_w, meta.ori_h, 
            meta.input_w, meta.input_h, 
            meta.pad_x, meta.pad_y
        )
        box.mask = mask_tensor
        box.mask_coeffs = nil
    end
    
    return final_boxes
end

return Model
