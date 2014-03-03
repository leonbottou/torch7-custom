
__global__ void DropmapKernel(float* idata, float* odata, float* maskdata, int iw, int str0, int str1, int str2, int ip)
{
   /* blockIdx.z  = [ 0, bs    ] ()
      blockIdx.x  = [ 0, ceil(ip/32) ] ()
      threadIdx.x = [ 0, 31    ] ()
      threadIdx.y = [ 0, ih    ] ()
   */

   const int planeidx=blockIdx.x*blockDim.x+threadIdx.x;
   idata += blockIdx.z*str0 + threadIdx.y*str1;
   odata += blockIdx.z*str0 + threadIdx.y*str1;

   __shared__ float mask[32];
   float localmask=0;

   if(planeidx<ip)
   {
      if (threadIdx.y==0)
      {
         localmask=maskdata[blockIdx.z*ip+planeidx];
         mask[threadIdx.x]=localmask;
      }
   }   

   __syncthreads();

   if (threadIdx.y>0)
   {
      localmask=mask[threadIdx.x];
   }
   
   int w;
   float v;

   if(planeidx<ip)
   {
      for (w=0; w<iw; w++)
      {
         v = idata[w*str2+planeidx];
         v = (localmask==1) ? v : 0;
         odata[w*str2+planeidx] = v;
      }
   }
   
}


__global__ void DropmapKernelSame(float* idata, float* odata, float* maskdata, int iw, int str0, int str1, int str2, int ip)
{
   /* blockIdx.z  = [ 0, bs    ] ()
      blockIdx.x  = [ 0, ceil(ip/32) ] ()
      threadIdx.x = [ 0, 31    ] ()
      threadIdx.y = [ 0, ih    ] ()
   */

   const int planeidx=blockIdx.x*blockDim.x+threadIdx.x;
   idata += blockIdx.z*str0 + threadIdx.y*str1;
   odata += blockIdx.z*str0 + threadIdx.y*str1;

   __shared__ float mask[32];
   float localmask=0;

   if(planeidx<ip)
   {
      if (threadIdx.y==0)
      {
         localmask=maskdata[planeidx];
         mask[threadIdx.x]=localmask;
      }
   }   

   __syncthreads();

   if (threadIdx.y>0)
   {
      localmask=mask[threadIdx.x];
   }
   
   int w;
   float v;

   if(planeidx<ip)
   {
      for (w=0; w<iw; w++)
      {
         v = idata[w*str2+planeidx];
         v = (localmask==1) ? v : 0;
         odata[w*str2+planeidx] = v;
      }
   }
   
}


static int cunxn_Dropmap_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *mask = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "mask", "torch.CudaTensor");
  
  float sameoverbatch = luaT_getfieldchecknumber(L, 1, "sameoverbatch");

  THCudaTensor_resizeAs(output, input);

  int bs = input->size[0];
  int ih = input->size[1];
  int iw = input->size[2];
  int ip = input->size[3];

  int str0 = input->stride[0];
  int str1 = input->stride[1];
  int str2 = input->stride[2];


  float* idata=THCudaTensor_data(input);
  float* odata=THCudaTensor_data(output);
  float* maskdata=THCudaTensor_data(mask);

  dim3 blocks((ip+31)/32, 1, bs);
  dim3 threads(32,ih);
  
  if(sameoverbatch==1)
  {
     DropmapKernelSame <<<blocks,threads>>>(idata, odata, maskdata, iw, str0, str1, str2, ip);
  }
  else 
  {
     DropmapKernel <<<blocks,threads>>>(idata, odata, maskdata, iw, str0, str1, str2, ip);
  }

  return 1;
}

static int cunxn_Dropmap_updateGradInput(lua_State *L)
{
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  THCudaTensor *mask = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "mask", "torch.CudaTensor");

  float sameoverbatch = luaT_getfieldchecknumber(L, 1, "sameoverbatch");
  
  THCudaTensor_resizeAs(gradInput, gradOutput);

  int bs = gradOutput->size[0];
  int ih = gradOutput->size[1];
  int iw = gradOutput->size[2];
  int ip = gradOutput->size[3];

  int str0 = gradOutput->stride[0];
  int str1 = gradOutput->stride[1];
  int str2 = gradOutput->stride[2];


  float* godata=THCudaTensor_data(gradOutput);
  float* gidata=THCudaTensor_data(gradInput);
  float* maskdata=THCudaTensor_data(mask);

  dim3 blocks((ip+31)/32, 1, bs);
  dim3 threads(32,ih);
  
  if(sameoverbatch==1)
  {
     DropmapKernelSame <<<blocks,threads>>> (godata, gidata, maskdata, iw, str0, str1, str2, ip);
  }
  else 
  {
     DropmapKernel <<<blocks,threads>>> (godata, gidata, maskdata, iw, str0, str1, str2, ip);
  }

  return 1;
}

static const struct luaL_Reg cunxn_Dropmap__ [] = {
  {"Dropmap_updateOutput", cunxn_Dropmap_updateOutput},
  {"Dropmap_updateGradInput", cunxn_Dropmap_updateGradInput},
  {NULL, NULL}
};

static void cunxn_Dropmap_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_ReLU__, "nxn");
  lua_pop(L,1);
}
