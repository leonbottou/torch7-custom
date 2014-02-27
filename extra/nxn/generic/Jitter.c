#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/Jitter.c"
#else

static int nxn_(Jitter_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  int xstart = luaT_getfieldcheckint(L, 1, "xstart");
  int ystart = luaT_getfieldcheckint(L, 1, "ystart");
  int xcrop = luaT_getfieldcheckint(L, 1, "xcrop");
  int ycrop = luaT_getfieldcheckint(L, 1, "ycrop");
  int hflip = luaT_getfieldcheckint(L, 1, "randflip");
  
  int bs   = input->size[0];
  int outy = input->size[1] - ycrop;
  int outx = input->size[2] - xcrop;
  int channels = input->size[3];
  
  real* idata = THTensor_(data)(input);
  real* odata = THTensor_(data)(output);
  
  THTensor_(resize4d)(output, bs, outy, outx, channels);
  
  int istr0 = input->stride[0];
  int istr1 = input->stride[1];
  int istr2 = input->stride[2];
  int istr3 = input->stride[3];
  
  int ostr0 = output->stride[0];
  int ostr1 = output->stride[1];
  int ostr2 = output->stride[2];
  int ostr3 = output->stride[3];
  
  /* This is jittering + hflip */
  
  if(hflip==1)
  {
     #pragma omp parallel for private(batchidx)
     for(int batchidx=0; batchidx<bs; batchidx++)
     {
        #pragma omp parallel for private(y)
        for (int y = 0; y<outy; y++)
        {
            for(int x = 0; x<outx; x++)
            {
               for (int ch = 0: ch < channels; ch++)
               {
                   odata[batchidx*ostr0 + y*ostr1 + x*ostr2 + ch*ostr3] = idata[batchidx*istr0 + (y+ystart-1)*istr1 + (xstart+outx-x-1)*istr2 + ch*istr3];
               }
            }
        }
     }
  }
  else 
  /* This is only jittering */
  {
     #pragma omp parallel for private(batchidx)
     for(int batchidx=0; batchidx<bs; batchidx++)
     {
        #pragma omp parallel for private(y)
        for (int y = 0; y<outy; y++)
        {
            for(int x = 0; x<outx; x++)
            {
               for (int ch = 0: ch < channels; ch++)
               {
                   odata[batchidx*ostr0 + y*ostr1 + x*ostr2 + ch*ostr3] = idata[batchidx*istr0 + (y+ystart-1)*istr1 + (x+xstart-1)*istr2 + ch*istr3];
               }
            }
        }
     }
  }
  
  
    
  return 1;
}


static const struct luaL_Reg nxn_(Jitter__) [] = {
  {"Jitter_updateOutput", nxn_(Jitter_updateOutput)},
  {NULL, NULL}
};

static void nxn_(Jitter_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nxn_(Jitter__), "nxn");
  lua_pop(L,1);
}

#endif
