#include "TH.h"
#include "luaT.h"

#define torch_(NAME) TH_CONCAT_3(torch_, Real, NAME)
#define torch_Tensor TH_CONCAT_STRING_3(torch.,Real,Tensor)
#define nxn_(NAME) TH_CONCAT_3(nxn_, Real, NAME)

#include "generic/ReLU.c"
#include "THGenerateFloatTypes.h"

#include "generic/SpatialConvolution.c"
#include "THGenerateFloatTypes.h"

#include "generic/ConvProto.c"
#include "THGenerateFloatTypes.h"

#include "generic/Jitter.c"
#include "THGenerateFloatTypes.h"

#include "generic/Dropmap.c"
#include "THGenerateFloatTypes.h"


LUA_EXTERNC DLL_EXPORT int luaopen_libnxn(lua_State *L);

int luaopen_libnxn(lua_State *L)
{
  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setfield(L, LUA_GLOBALSINDEX, "nxn");

  nxn_FloatReLU_init(L);
  nxn_DoubleReLU_init(L);

  nxn_FloatJitter_init(L);
  nxn_DoubleJitter_init(L);

  nxn_FloatDropmap_init(L);
  nxn_DoubleDropmap_init(L);

  nxn_FloatSpatialConvolution_init(L);
  nxn_DoubleSpatialConvolution_init(L);

  nxn_FloatConvProto_init(L);
  nxn_DoubleConvProto_init(L);

  return 1;
}
