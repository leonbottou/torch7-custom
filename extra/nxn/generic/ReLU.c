#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ReLU.c"
#else

static int nxn_(ReLU_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  
  THTensor_(resizeAs)(output, input);

  TH_TENSOR_APPLY2(real, output, real, input,         \
                   *output_data = *input_data > 0 ? *input_data : 0;)
    
  return 1;
}

static int nxn_(ReLU_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);

  THTensor_(resizeAs)(gradInput, output);
  TH_TENSOR_APPLY3(real, gradInput, real, gradOutput, real, output,     \
                   *gradInput_data = *gradOutput_data * (*output_data > 0 ? 1 : 0););
  return 1;
}

static const struct luaL_Reg nxn_(ReLU__) [] = {
  {"ReLU_updateOutput", nxn_(ReLU_updateOutput)},
  {"ReLU_updateGradInput", nxn_(ReLU_updateGradInput)},
  {NULL, NULL}
};

static void nxn_(ReLU_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nxn_(ReLU__), "nxn");
  lua_pop(L,1);
}

#endif
