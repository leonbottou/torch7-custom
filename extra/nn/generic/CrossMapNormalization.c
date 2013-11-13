#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CrossMapNormalization.c"
#else

static int nn_(CrossMapNormalization_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  real alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  real beta = luaT_getfieldchecknumber(L, 1, "beta");
  real k = luaT_getfieldchecknumber(L, 1, "k");
  long n = luaT_getfieldcheckint(L, 1, "n");
  real alphan = alpha / n;
  long i,j;

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");
  
  THTensor_(resizeAs)(output, input);
  
  TH_TENSOR_DIM_APPLY2(real, output, real, input, dimension,
                       for(i = 0; i < input_size; i++)
                       {
                         real z = 0;
                         long startf = i - n/2;
                         long endf = startf + n;
                         startf = (startf < 0) ? 0 : startf;
                         endf = (endf > input_size) ? input_size : endf;
                         for(j=startf; j<endf; j++)
                           {
                             real x = input_data[j*input_stride];
                             z += x * x;
                           }
                         z=k+z*alphan;
                         output_data[i*output_stride] = input_data[i*input_stride] * pow(z, -beta);
                       }
                      );
  return 1;
}

static int nn_(CrossMapNormalization_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dimension  = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THTensor *gradInput  = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  real alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  real beta = luaT_getfieldchecknumber(L, 1, "beta");
  real k = luaT_getfieldchecknumber(L, 1, "k");
  long n = luaT_getfieldcheckint(L, 1, "n");
  real alphan = alpha / n;
  long i, j, m;
  real *zi0, *zi1;

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");
  zi0 = malloc(sizeof(real)*input->size[dimension]);
  zi1 = malloc(sizeof(real)*input->size[dimension]);
  luaL_argcheck(L, zi0 != 0 && zi1 != 0, 1, "out of memory");
              
  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);

  TH_TENSOR_DIM_APPLY3(real, gradInput, real, input, real, gradOutput, dimension,
                       for(i = 0; i < input_size; i++)
                       {
                         real z = 0;
                         long startf = i - n/2;
                         long endf = startf + n;
                         startf = (startf < 0) ? 0 : startf;
                         endf = (endf > input_size) ? input_size : endf;
                         for(j=startf; j<endf; j++)
                           {
                             real x = input_data[j*input_stride];
                             z += x * x;
                           }
                         z=k+z*alphan;
                         zi0[i] = pow(z, -beta);  /* z^{-beta}   */
                         zi1[i] = zi0[i] / z;     /* z^{-beta-1} */
                       }
                      for(i = 0; i < input_size; i++)
                       {
                         real gradi = 0;
                         real ai = input_data[i*input_stride];
                         long endo = i + n/2 + 1;
                         long starto = endo - n;
                         starto = (starto < 0) ? 0 : starto;
                         endo = (endo > input_size) ? input_size : endo;
                         for (j=starto; j<endo; j++)
                           {
                             real aj = input_data[j*input_stride];
                             real gj = gradOutput_data[j*gradOutput_stride];
                             gradi += (i == j) ? gj * zi0[j] : 0;
                             gradi -= gj * 2 * alphan * beta * ai * aj * zi1[j];
                           }
                         gradInput_data[i*gradInput_stride]=gradi;
                       }
                      );
  
  free(zi0);
  free(zi1);
  return 1;
}

static const struct luaL_Reg nn_(CrossMapNormalization__) [] = {
  {"CrossMapNormalization_updateOutput", nn_(CrossMapNormalization_updateOutput)},
  {"CrossMapNormalization_updateGradInput", nn_(CrossMapNormalization_updateGradInput)},
  {NULL, NULL}
};

static void nn_(CrossMapNormalization_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(CrossMapNormalization__), "nn");
  lua_pop(L,1);
}

#endif
