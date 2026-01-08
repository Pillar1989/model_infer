-- Common Preprocessing Functions for YOLO Models

local M = {}

-- Letterbox预处理：缩放+padding
-- 保持宽高比的同时将图片缩放到目标大小
-- 参数:
--   img: Image对象
--   input_size: {height, width} 目标尺寸
--   stride: padding对齐步长 (默认32)
--   fill_value: padding填充值 (默认114)
-- 返回:
--   input_tensor: 预处理后的tensor
--   meta: 元数据 {scale, pad_x, pad_y, ori_w, ori_h}
function M.letterbox(img, input_size, stride, fill_value)
    stride = stride or 32
    fill_value = fill_value or 114

    local w, h = img.width, img.height
    local target_h, target_w = table.unpack(input_size)

    -- 计算缩放比例（保持宽高比）
    local r = math.min(target_h / h, target_w / w)
    local new_w = math.floor(w * r)
    local new_h = math.floor(h * r)

    -- 缩放图片
    if new_w ~= w or new_h ~= h then
        img:resize(new_w, new_h)
    end

    -- 计算padding（对齐到stride）
    local dw = target_w - new_w
    local dh = target_h - new_h

    dw = dw % stride
    dh = dh % stride

    local top = math.floor(dh / 2)
    local bottom = dh - top
    local left = math.floor(dw / 2)
    local right = dw - left

    -- 添加padding
    img:pad(top, bottom, left, right, fill_value)

    -- 转换为tensor
    local scale = 1.0 / 255.0
    local input_tensor = img:to_tensor(scale, {0,0,0}, {1,1,1})

    -- 返回元数据用于后处理坐标转换
    local meta = {
        scale = r,
        pad_x = left,
        pad_y = top,
        ori_w = w,
        ori_h = h
    }

    return input_tensor, meta
end

-- 坐标缩放和去padding
-- 将模型输出的坐标转换回原始图片坐标
function M.scale_coords(x, y, meta)
    local scaled_x = (x - meta.pad_x) / meta.scale
    local scaled_y = (y - meta.pad_y) / meta.scale
    return scaled_x, scaled_y
end

-- 尺寸缩放（用于width/height）
function M.scale_size(size, meta)
    return size / meta.scale
end

return M
