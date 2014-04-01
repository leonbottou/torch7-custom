#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/CrossMapNormalization.c"
#else

#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

static int nxn_(CrossMapNormalization_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  int dimension = 4-1;
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *zsave = luaT_getfieldcheckudata(L, 1, "z", torch_Tensor);
  real alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  real beta = luaT_getfieldchecknumber(L, 1, "beta");
  real k = luaT_getfieldchecknumber(L, 1, "k");
  long n = luaT_getfieldcheckint(L, 1, "n");
  real alphan = alpha / n;
  
  long bs=input->size[0];
  long isize1=input->size[1];
  long isize2=input->size[2];
  long npix=bs*isize1*isize2;
  long planes=input->size[3];
  long istr2=input->stride[2];
  assert(istr2==planes);

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");
  
  THTensor_(resizeAs)(output, input);
  THTensor_(resizeAs)(zsave, input);
  
/*  TH_TENSOR_DIM_APPLY2(real, output, real, input, dimension,
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
                      );*/

  real * inptr = THTensor_(data)(input);
  real * zptr = THTensor_(data)(zsave);
  real * optr = THTensor_(data)(output);
  
  
  long idx;
  #pragma omp parallel for private(idx)
  for(idx=0; idx<npix; idx++)
  {
     long ch;
     long j;
     real * curinptr=inptr+idx*istr2;
     real * curzptr=zptr+idx*planes;
     real * curoptr=optr+idx*planes;
     for(ch=0; ch<planes; ch++)
     {
         real z=0;
         real val;
         long startf = ch - n/2;
         long endf = startf + n;
         startf = (startf < 0) ? 0 : startf;
         endf = (endf > planes) ? planes : endf;
         for(j=startf; j<endf; j++)
         {
             real x = curinptr[j];
             z += x * x;
             if (j==ch) val=x;
         }
         z=k+z*alphan;
         curzptr[ch]=z;
         curoptr[ch]=val*pow(z, -beta);
     }         
  }


  return 1;
}

static int nxn_(CrossMapNormalization_updateGradInput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *gradInput  = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  THTensor *zsave  = luaT_getfieldcheckudata(L, 1, "z", torch_Tensor);
  real alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  real beta = luaT_getfieldchecknumber(L, 1, "beta");
  real k = luaT_getfieldchecknumber(L, 1, "k");
  long n = luaT_getfieldcheckint(L, 1, "n");
  real alphan = alpha / n;

  long bs=input->size[0];
  long isize1=input->size[1];
  long isize2=input->size[2];
  long npix=bs*isize1*isize2;
  long planes=input->size[3];
  long istr2=input->stride[2];
  long gostr2=gradOutput->stride[2];

  THTensor_(resizeAs)(gradInput, input);
  THTensor_(zero)(gradInput);
  
  
  real * inptr = THTensor_(data)(input);
  real * zptr = THTensor_(data)(zsave);
  real * goptr = THTensor_(data)(gradOutput);
  real * giptr = THTensor_(data)(gradInput);

  long idx;
  #pragma omp parallel for private(idx)
  for(idx=0; idx<npix; idx++)
  {
     long ch;
     long j;
     real * curinptr=inptr+idx*istr2;
     real * curzptr=zptr+idx*planes;
     real * curgoptr=goptr+idx*gostr2;
     real * curgiptr=giptr+idx*planes;
     /*for(ch=0; ch<planes; ch++)
     {
         real gradi = 0;
         real ai = curinptr[ch];
         long endo = ch + n/2 + 1;
         long starto = endo - n;
         starto = (starto < 0) ? 0 : starto;
         endo = (endo > planes) ? planes : endo;
         for (j=starto; j<endo; j++)
         {
            real aj = curinptr[j];
            real gj = curgoptr[j];
            gradi += (ch == j) ? gj * pow(curzptr[j], -beta) : 0;
            gradi -= gj * 2 * alphan * beta * ai * aj * pow(curzptr[j], -beta-1);
         }
         curgiptr[ch]=gradi;
     }*/
     for(ch=0; ch<planes; ch++)
     {
         real z   = curzptr[ch];
         real zb  = pow(curzptr[ch], -beta);
         real zb2 = zb/z;
         
         real gj = curgoptr[ch];
         real aj = curinptr[ch];
         curgiptr[ch] =gj*zb;
         curzptr[ch] = gj * 2 * alphan * beta * aj * zb2;
     }
     for(ch=0; ch<planes; ch++)
     {
         real ai = curinptr[ch];
         long endo = ch + n/2 + 1;
         long starto = endo - n;
         starto = (starto < 0) ? 0 : starto;
         endo = (endo > planes) ? planes : endo;
         for (j=starto; j<endo; j++)
         {
             curgiptr[ch] -= ai * curzptr[j];
         }         
     }     
  }
  
  
  
  
  
  
  
  
  return 1;
}

static const struct luaL_Reg nxn_(CrossMapNormalization__) [] = {
  {"CrossMapNormalization_updateOutput", nxn_(CrossMapNormalization_updateOutput)},
  {"CrossMapNormalization_updateGradInput", nxn_(CrossMapNormalization_updateGradInput)},
  {NULL, NULL}
};

static void nxn_(CrossMapNormalization_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nxn_(CrossMapNormalization__), "nxn");
  lua_pop(L,1);
}

#endif
