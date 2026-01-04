-- Import our defined C++ modules (this line can be omitted if C++ is globally registered)
-- local cv = require "lua_cv"
-- local nn = require "lua_nn"
-- local utils = require "lua_utils"

local Model = {}

-- ==========================================================
-- 1. Model Configuration (Config)
-- ==========================================================
Model.config = {
    input_size = {640, 640},  -- [H, W]
    conf_thres = 0.25,
    iou_thres  = 0.45,
    stride     = 32,
    -- COCO class labels
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

-- ==========================================================
-- 2. Pre-processing
-- Input: lua_cv.Image object
-- Output: lua_nn.Tensor object, meta information table
-- ==========================================================
function Model.preprocess(img)
    -- 1. Get original dimensions
    local w, h = img.width, img.height
    local target_h, target_w = table.unpack(Model.config.input_size)

    -- 2. Calculate scaling ratio (Letterbox logic)
    local r = math.min(target_h / h, target_w / w)
    
    -- 3. Calculate new dimensions
    local new_w = math.floor(w * r)
    local new_h = math.floor(h * r)

    -- 4. Image resizing (call lua_cv)
    if new_w ~= w or new_h ~= h then
        img:resize(new_w, new_h) -- Default linear interpolation
    end

    -- 5. Calculate Padding
    local dw = target_w - new_w
    local dh = target_h - new_h
    
    -- Ensure stride alignment
    dw = dw % Model.config.stride
    dh = dh % Model.config.stride

    -- Center calculation
    local top = math.floor(dh / 2)
    local bottom = dh - top
    local left = math.floor(dw / 2)
    local right = dw - left

    -- 6. Border padding (call lua_cv)
    -- Parameters: top, bottom, left, right, fill_value
    img:pad(top, bottom, left, right, 114)

    -- 7. Convert to Tensor (HWC->CHW, Normalize)
    -- YOLOv5/v8/v11 usually needs division by 255 (scale=1/255), mean=0, std=1
    local scale = 1.0 / 255.0
    local input_tensor = img:to_tensor(scale, {0,0,0}, {1,1,1})

    -- 8. Record Meta information for restoration
    local meta = {
        scale = r,
        pad_x = left,
        pad_y = top,
        ori_w = w,
        ori_h = h
    }

    return input_tensor, meta
end

-- ==========================================================
-- 3. Post-processing
-- Input: Inference result Map, meta information
-- Output: Final detection box list
-- ==========================================================
function Model.postprocess(outputs, meta)
    -- Get output Tensor
    local output_tensor = outputs["output0"] -- Assume model output node name is output0

    -- STEP 1: Quick filtering (Critical Optimization)
    -- Call C++ implemented filter to avoid Lua looping 25200 times
    -- Return format: { {x,y,w,h,score,class_id}, ... } in original model coordinates
    -- Note: C++ implementation now handles both YOLOv5 [1, N, 85] and YOLOv8/11 [1, 84, N] formats automatically
    local raw_boxes = output_tensor:filter_yolo(Model.config.conf_thres)

    local proposals = {}

    -- STEP 2: Coordinate restoration (only process the small number of filtered boxes)
    for _, box in ipairs(raw_boxes) do
        -- Reverse Letterbox calculation
        -- Box structure defined by C++: box.x, box.y, box.w, box.h, box.score, box.cls
        
        local x = (box.x - meta.pad_x) / meta.scale
        local y = (box.y - meta.pad_y) / meta.scale
        local w = box.w / meta.scale
        local h = box.h / meta.scale

        -- Boundary clipping (optional, prevent boxes from exceeding original image)
        x = math.max(0, x)
        y = math.max(0, y)
        w = math.min(w, meta.ori_w - x)
        h = math.min(h, meta.ori_h - y)

        table.insert(proposals, {
            x = x, y = y, w = w, h = h,
            score = box.score,
            label = Model.config.labels[box.cls + 1] -- Lua index starts from 1
        })
    end

    -- STEP 3: NMS (call lua_utils)
    local final_results = lua_utils.nms(proposals, Model.config.iou_thres)

    return final_results
end

return Model
