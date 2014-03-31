#include "luaT.h"
#include "THC.h"
#include "THLogAdd.h" /* DEBUG: WTF */

#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>

#include "ReLU.cu"
#include "CrossMapNormalization.cu"
#include "SpatialConvolution.cu"
#include "SpatialMaxPooling.cu"
#include "SpatialGlobalMaxPooling.cu"
#include "testSgemm.cu"
#include "ConvProto.cu"
#include "Dropmap.cu"
#include "LogSoftMax.cu"


LUA_EXTERNC DLL_EXPORT int luaopen_libcunxn(lua_State *L);

int luaopen_libcunxn(lua_State *L)
{
  lua_newtable(L);

  cunxn_ReLU_init(L);

  cunxn_CrossMapNormalization_init(L);
  cunxn_SpatialConvolution_init(L);
  cunxn_SpatialMaxPooling_init(L);
  cunxn_SpatialGlobalMaxPooling_init(L);
  cunxn_testSgemm_init(L);
  cunxn_ConvProto_init(L);
  cunxn_Dropmap_init(L);
  cunxn_LogSoftMax_init(L);

  return 1;
}
