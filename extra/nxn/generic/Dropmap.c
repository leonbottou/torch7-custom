#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Dropmap.c"
#else

static int nxn_(Dropmap_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *mask = luaT_getfieldcheckudata(L, 1, "mask", torch_Tensor);
  
  float sameoverbatch = luaT_getfieldchecknumber(L, 1, "sameoverbatch");
    
  int bs   = input->size[0];
  int ymax = input->size[1];
  int xmax = input->size[2];
  int channels = input->size[3];
  
  THTensor_(resizeAs)(output, input);

  real* idata = THTensor_(data)(input);
  real* odata = THTensor_(data)(output);
  real* maskdata = THTensor_(data)(mask);

  int istr0 = input->stride[0];
  int istr1 = input->stride[1];
  int istr2 = input->stride[2];
  int istr3 = input->stride[3];
  
  int ostr0 = output->stride[0];
  int ostr1 = output->stride[1];
  int ostr2 = output->stride[2];
  int ostr3 = output->stride[3];
  
  int batchidx, y, x, ch;
  
  if(sameoverbatch==1)
  {
     #pragma omp parallel for private(batchidx)
     for(batchidx=0; batchidx<bs; batchidx++)
     {
        for (y = 0; y<ymax; y++)
        {
           for(x = 0; x<xmax; x++)
           {
              for (ch = 0; ch < channels; ch++)
              {
                 if(maskdata[ch]==0) { odata[batchidx*ostr0 + y*ostr1 + x*ostr2 + ch*ostr3] = 0; }
                 else                { odata[batchidx*ostr0 + y*ostr1 + x*ostr2 + ch*ostr3] = idata[batchidx*istr0 + y*istr1 + x*istr2 + ch*istr3]; }
              }
           }
        }
     }
  }
  else
  {
     #pragma omp parallel for private(batchidx)
     for(batchidx=0; batchidx<bs; batchidx++)
     {
        for (y = 0; y<ymax; y++)
        {
           for(x = 0; x<xmax; x++)
           {
              for (ch = 0; ch < channels; ch++)
              {
                 if(maskdata[batchidx*channels+ch]==0) { odata[batchidx*ostr0 + y*ostr1 + x*ostr2 + ch*ostr3] = 0; }
                 else                                  { odata[batchidx*ostr0 + y*ostr1 + x*ostr2 + ch*ostr3] = idata[batchidx*istr0 + y*istr1 + x*istr2 + ch*istr3]; }
              }
           }
        }
     }
  }

  return 1;
}

static int nxn_(Dropmap_updateGradInput)(lua_State *L)
{
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  THTensor *mask = luaT_getfieldcheckudata(L, 1, "mask", torch_Tensor);
  
  THTensor_(resizeAs)(gradInput, gradOutput);

  int bs   = gradOutput->size[0];
  int ymax = gradOutput->size[1];
  int xmax = gradOutput->size[2];
  int channels = gradOutput->size[3];

  real* gidata = THTensor_(data)(gradInput);
  real* godata = THTensor_(data)(gradOutput);
  real* maskdata = THTensor_(data)(mask);

  int gistr0 = gradInput->stride[0];
  int gistr1 = gradInput->stride[1];
  int gistr2 = gradInput->stride[2];
  int gistr3 = gradInput->stride[3];
  
  int gostr0 = gradOutput->stride[0];
  int gostr1 = gradOutput->stride[1];
  int gostr2 = gradOutput->stride[2];
  int gostr3 = gradOutput->stride[3];

  int batchidx, y, x, ch;
  
  if(sameoverbatch==1)
  {
     #pragma omp parallel for private(batchidx)
     for(batchidx=0; batchidx<bs; batchidx++)
     {
        for (y = 0; y<ymax; y++)
        {
           for(x = 0; x<xmax; x++)
           {
              for (ch = 0; ch < channels; ch++)
              {
                 if(maskdata[ch]==0) { gidata[batchidx*gistr0 + y*gistr1 + x*gistr2 + ch*gistr3] = 0; }
                 else                { gidata[batchidx*gistr0 + y*gistr1 + x*gistr2 + ch*gistr3] = godata[batchidx*gostr0 + y*gostr1 + x*gostr2 + ch*gostr3]; }
              }
           }
        }
     }
  }
  else
  {
     #pragma omp parallel for private(batchidx)
     for(batchidx=0; batchidx<bs; batchidx++)
     {
        for (y = 0; y<ymax; y++)
        {
           for(x = 0; x<xmax; x++)
           {
              for (ch = 0; ch < channels; ch++)
              {
                 if(maskdata[batchidx*channels+ch]==0) { gidata[batchidx*gistr0 + y*gistr1 + x*gistr2 + ch*gistr3] = 0; }
                 else                { gidata[batchidx*gistr0 + y*gistr1 + x*gistr2 + ch*gistr3] = godata[batchidx*gostr0 + y*gostr1 + x*gostr2 + ch*gostr3]; }
                  
              }
           }
        }
     }
  }
  
  return 1;
}

static const struct luaL_Reg nxn_(Dropmap__) [] = {
  {"ReLU_updateOutput", nxn_(Dropmap_updateOutput)},
  {"ReLU_updateGradInput", nxn_(Dropmap_updateGradInput)},
  {NULL, NULL}
};

static void nxn_(Dropmap_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nxn_(Dropmap__), "nxn");
  lua_pop(L,1);
}

#endif
