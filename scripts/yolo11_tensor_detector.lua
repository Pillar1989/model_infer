-- YOLO11 Object Detection Script (使用新Tensor API)
-- 展示如何用通用Tensor操作替代filter_yolo()等专用方法
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
        ori_h = h
    }

    return input_tensor, meta
end

-- 使用通用Tensor操作实现YOLO后处理
function Model.postprocess(outputs, meta)
    local output = outputs["output0"]
    
    if not output then 
        error("Missing output0") 
    end

    -- YOLO11格式: [1, 84, 8400]
    -- 前4个是box坐标(xywh), 后80个是类别分数
    -- 直接使用已知格式，不依赖shape()方法
    local num_classes = 80
    
    -- 1. 提取box坐标和类别分数
    -- boxes: [1, 4, 8400] - 前4行
    local boxes = output:slice(1, 0, 4, 1):squeeze(0)  -- squeeze掉batch维度 -> [4, 8400]
    
    -- scores: [1, 80, 8400] - 后80行
    local scores = output:slice(1, 4, 84, 1):squeeze(0)  -- -> [80, 8400]
    
    -- 2. 对每个box找到最大分数和对应类别
    -- max_scores: [8400], class_ids: Lua table [8400]
    local max_scores = scores:max(0, false)  -- 沿类别维度取最大值，不保持维度
    local class_ids = scores:argmax(0)  -- 获取最大值的索引，直接返回Lua table
    
    -- 3. 过滤低置信度的box
    -- 转为table进行过滤(后续可优化为纯Tensor操作)
    local boxes_table = boxes:to_table()  -- [4][N]
    local max_scores_table = max_scores:to_table()  -- [N]
    -- class_ids已经是table了
    
    -- 使用实际长度（考虑到不同输入尺寸可能产生不同数量的anchor）
    local actual_num_boxes = #max_scores_table
    
    local proposals = {}
    
    for i = 1, actual_num_boxes do
        -- 访问1D和2D table
        local conf = max_scores_table[i]
        
        if conf >= Model.config.conf_thres then
            local cx = boxes_table[1][i]
            local cy = boxes_table[2][i]
            local w = boxes_table[3][i]
            local h = boxes_table[4][i]
            local cls_id = class_ids[i]  -- class_ids已经是table
            
            -- 将中心点坐标转换为左上角坐标
            local x = cx - w / 2.0
            local y = cy - h / 2.0
            
            table.insert(proposals, {
                x = x,
                y = y,
                w = w,
                h = h,
                score = conf,
                class_id = cls_id,
                label = Model.config.labels[cls_id + 1] or "unknown"
            })
        end
    end
    
    print(string.format("过滤后候选框: %d", #proposals))
    
    -- 4. 坐标还原到原图
    for _, box in ipairs(proposals) do
        box.x = (box.x - meta.pad_x) / meta.scale
        box.y = (box.y - meta.pad_y) / meta.scale
        box.w = box.w / meta.scale
        box.h = box.h / meta.scale
    end
    
    -- 5. NMS
    local final_boxes = utils.nms(proposals, Model.config.iou_thres)
    
    print(string.format("NMS后最终框: %d", #final_boxes))
    
    return final_boxes
end

return Model
