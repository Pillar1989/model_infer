#ifndef MODEL_INFER_LUA_UTILS_H_
#define MODEL_INFER_LUA_UTILS_H_

#include "LuaIntf.h"
#include <vector>

namespace lua_utils {

struct Box {
    float x, y, w, h;
    float score;
    int label;
};

// NMS算法
LuaIntf::LuaRef nms(LuaIntf::LuaRef proposals, float iou_thres);

// 辅助函数
float compute_iou(const Box& a, const Box& b);

// 注册到Lua
void register_module(lua_State* L);

} // namespace lua_utils

#endif
