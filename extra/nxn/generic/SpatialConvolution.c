#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolution.c"
#else

/* -------------------------------------- */
/* Generic convolution routines           */
/* -------------------------------------- */






/* -------------------------------------- */
/* Torch nxn wrappers                     */
/* -------------------------------------- */


static int nxn_(SpatialConvolution_updateOutput)(lua_State *L)
{
#if 0
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int dimw = 2;
  int dimh = 1;
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");
#endif

  luaL_error(L, "not implemented");
  return 0;
}


static int nxn_(SpatialConvolution_updateGradInput)(lua_State *L)
{
#if 0
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  THTensor *tweight;
  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );
#endif

  luaL_error(L, "not implemented");
  return 0;
}


static int nxn_(SpatialConvolution_accGradParameters)(lua_State *L)
{
#if 0
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  int dimw = 2;
  int dimh = 1;
  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );
#endif

  luaL_error(L, "not implemented");
  return 0;
}

static const struct luaL_Reg nxn_(SpatialConvolution__) [] = {
  {"SpatialConvolution_updateOutput", nxn_(SpatialConvolution_updateOutput)},
  {"SpatialConvolution_updateGradInput", nxn_(SpatialConvolution_updateGradInput)},
  {"SpatialConvolution_accGradParameters", nxn_(SpatialConvolution_accGradParameters)},
  {NULL, NULL}
};

static void nxn_(SpatialConvolution_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nxn_(SpatialConvolution__), "nxn");
  lua_pop(L,1);
}

#endif
