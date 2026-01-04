#include "../modules/lua_cv.h"
#include "../modules/lua_nn.h"
#include "../modules/lua_utils.h"

void register_all_modules(lua_State* L) {
    lua_cv::register_module(L);
    lua_nn::register_module(L);
    lua_utils::register_module(L);
}
