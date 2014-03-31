#define MIN(a,b) (a) < (b) ? (a) : (b)
#define MAX(a,b) (a) > (b) ? (a) : (b)


__global__ void DummyKernel(float* idata, float* odata, int numclasses)
{
int foo=threadIdx.x;
__shfl(foo, 0);

}


__global__ void LogSoftMaxForwardKernel(float* idata, float* odata, int numclasses)
{
   /*blockIdx.z = [0, bs] 
     threadIdx.x= [0, 31]
   */
   
   __shared__ volatile float s[32];
   
   idata += blockIdx.z*numclasses;
   odata += blockIdx.z*numclasses;
   
   /* compute max */
   float maxInput=-2e38;
   
   for(int d=threadIdx.x; d<numclasses; d+=blockDim.x)
   {
      maxInput=MAX(maxInput, idata[d]);
   }
   
   s[threadIdx.x]=maxInput;
   
   /* Y U NO __SHFL */
   if(threadIdx.x<16) { s[threadIdx.x] = MAX(s[threadIdx.x], s[threadIdx.x+16]); }
   if(threadIdx.x<8) { s[threadIdx.x] = MAX(s[threadIdx.x], s[threadIdx.x+8]); }
   if(threadIdx.x<4) { s[threadIdx.x] = MAX(s[threadIdx.x], s[threadIdx.x+4]); }
   if(threadIdx.x<2) { s[threadIdx.x] = MAX(s[threadIdx.x], s[threadIdx.x+2]); }
   if(threadIdx.x<1) { s[threadIdx.x] = MAX(s[threadIdx.x], s[threadIdx.x+1]); }

   if(threadIdx.x<1) { s[threadIdx.x+1] = s[threadIdx.x]; }
   if(threadIdx.x<2) { s[threadIdx.x+2] = s[threadIdx.x]; }
   if(threadIdx.x<4) { s[threadIdx.x+4] = s[threadIdx.x]; }
   if(threadIdx.x<8) { s[threadIdx.x+8] = s[threadIdx.x]; }
   if(threadIdx.x<16) { s[threadIdx.x+16] = s[threadIdx.x]; }

   maxInput=s[threadIdx.x];
   
   
   float logsum=0;
   
   /* compute logsum */
   
   for(int d=threadIdx.x; d<numclasses; d+=blockDim.x)
   {
      logsum += expf(idata[d]-maxInput);
   }

   s[threadIdx.x]=logsum;
   
   /* Y U NO __SHFL */
   if(threadIdx.x<16) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+16]; }
   if(threadIdx.x<8) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+8]; }
   if(threadIdx.x<4) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+4]; }
   if(threadIdx.x<2) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+2]; }
   if(threadIdx.x<1) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+1]; }
   
   if(threadIdx.x<1) { s[threadIdx.x+1] = s[threadIdx.x]; }
   if(threadIdx.x<2) { s[threadIdx.x+2] = s[threadIdx.x]; }
   if(threadIdx.x<4) { s[threadIdx.x+4] = s[threadIdx.x]; }
   if(threadIdx.x<8) { s[threadIdx.x+8] = s[threadIdx.x]; }
   if(threadIdx.x<16) { s[threadIdx.x+16] = s[threadIdx.x]; }

   logsum = maxInput + logf(s[threadIdx.x]);

   /* compute output */
   for(int d=threadIdx.x; d<numclasses; d+=blockDim.x)
   {
      odata[d]=idata[d] - logsum;
   }

}

__global__ void LogSoftMaxBackwardKernel(float* godata, float* gidata, float* odata, int numclasses)
{
   /*blockIdx.z = [0, bs] 
     threadIdx.x= [0, 31]
   */
   
   __shared__ volatile float s[32];
   
   odata += blockIdx.z*numclasses;
   godata += blockIdx.z*numclasses;
   gidata += blockIdx.z*numclasses;
   
   /* compute sum */
   float sum=0;
   
   for(int d=threadIdx.x; d<numclasses; d+=blockDim.x)
   {
      sum += godata[d]; 
   }
   
   s[threadIdx.x]=sum;
   __syncthreads();
   
   /* Y U NO __SHFL */
   if(threadIdx.x<16) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+16]; }
   //__syncthreads();
   if(threadIdx.x<8) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+8]; }
   //__syncthreads();
   if(threadIdx.x<4) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+4]; }
   //__syncthreads();
   if(threadIdx.x<2) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+2]; }
   //__syncthreads();
   if(threadIdx.x<1) { s[threadIdx.x] = s[threadIdx.x] + s[threadIdx.x+1]; }
   //__syncthreads();
   
   if(threadIdx.x<1) { s[threadIdx.x+1] = s[threadIdx.x]; }
   //__syncthreads();
   if(threadIdx.x<2) { s[threadIdx.x+2] = s[threadIdx.x]; }
   //__syncthreads();
   if(threadIdx.x<4) { s[threadIdx.x+4] = s[threadIdx.x]; }
   //__syncthreads();
   if(threadIdx.x<8) { s[threadIdx.x+8] = s[threadIdx.x]; }
   //__syncthreads();
   if(threadIdx.x<16) { s[threadIdx.x+16] = s[threadIdx.x]; }
   //__syncthreads();

   sum=s[threadIdx.x];
   
   /* compute gradinput */
   for(int d=threadIdx.x; d<numclasses; d+=blockDim.x)
   {
      gidata[d] = godata[d] - expf(odata[d])*sum;
   }
   
}



static int cunxn_LogSoftMax_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor*)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  if(input->nDimension != 2)
  {
    THArgCheck(0, 2, "matrix (batchsize*numclasses) expected");
  }

  int bs = input->size[0];
  int numclasses = input->size[1];
  
  THCudaTensor_resizeAs(output, input);
  
  float* idata=THCudaTensor_data(input);
  float* odata=THCudaTensor_data(output);

  dim3 blocks(1,1,bs);
  dim3 threads(32);
  
  LogSoftMaxForwardKernel<<<blocks, threads>>>(idata, odata, numclasses);

  return 1;
}

static int cunxn_LogSoftMax_updateGradInput(lua_State *L)
{
  THCudaTensor *gradOutput = (THCudaTensor*)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor*)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");

  if(output->nDimension != 2)
  {
    THArgCheck(0, 2, "matrix (batchsize*numclasses) expected");
  }

  int bs = output->size[0];
  int numclasses = output->size[1];
  
  THCudaTensor_resizeAs(gradInput, gradOutput);
  
  float* gidata=THCudaTensor_data(gradInput);
  float* godata=THCudaTensor_data(gradOutput);
  float* odata=THCudaTensor_data(output);

  dim3 blocks(1,1,bs);
  dim3 threads(32);
  
  LogSoftMaxBackwardKernel<<<blocks, threads>>>(godata, gidata, odata, numclasses);

  return 1;
}

static const struct luaL_Reg cunxn_LogSoftMax__ [] = {
  {"LogSoftMax_updateOutput", cunxn_LogSoftMax_updateOutput},
  {"LogSoftMax_updateGradInput", cunxn_LogSoftMax_updateGradInput},
  {NULL, NULL}
};

static void cunxn_LogSoftMax_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_LogSoftMax__, "nxn");
  lua_pop(L,1);
}
