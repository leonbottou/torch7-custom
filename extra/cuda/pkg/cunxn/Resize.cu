#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

static texture<float4, cudaTextureType2DLayered> texRef;

__global__ void resizeKernel(float* outptr, int outstr0, int outstr1, int outstr2, int outstr3, int outx)
{
   // blockIdx.x = 0, iw*scale-1
   // blockIdx.y = 0, ih*scale-1
   // blockIdx.z = 0, bs-1
   // threadIdx.x= 0, 2
   // threadIdx.y= 0, 31
   

   float coordx = (float)(blockIdx.x*32+threadIdx.y)/outx;
   float coordy = (float)blockIdx.y/(float)gridDim.y;
   if (coordx<1)
   {
      float4 out    = tex2DLayered(texRef, coordx, coordy, blockIdx.z);
      
      float val;
      if(threadIdx.x % 3 == 0) val=out.x;
      if(threadIdx.x % 3 == 1) val=out.y;
      if(threadIdx.x % 3 == 2) val=out.z;

      outptr[blockIdx.z*outstr0+blockIdx.y*outstr1+(blockIdx.x*32+threadIdx.y)*outstr2+threadIdx.x]=val;
   }

}

__global__ void resizeTiledKernel(float* outptr, int outstr0, int outstr1, int outstr2, int outstr3, int outx, int outy)
{
   // blockIdx.x = 0, iw*scale-1 / 8
   // blockIdx.y = 0, ih*scale-1 / 4
   // blockIdx.z = 0, bs-1
   // threadIdx.x= 0, 7
   // threadIdx.y= 0, 3
   

   float coordx = (float)(blockIdx.x*8+threadIdx.x)/outx;
   float coordy = (float)(blockIdx.y*4+threadIdx.y)/outy;
   int tidx = threadIdx.y*blockDim.x+threadIdx.x;
   float4 out;
   float ok=0;
   float ok2;
   __shared__ volatile float writevalues[32];
   if (coordx<1 && coordy<1)
   {
   // read :
      out    = tex2DLayered(texRef, coordx, coordy, blockIdx.z);
      ok=1;
   

   // spread one line :
   for (int ty=0; ty<4; ty++)
   {

      if (threadIdx.y==ty)
      {
         writevalues[threadIdx.x*3]=out.x;
         writevalues[threadIdx.x*3+1]=out.y;
         writevalues[threadIdx.x*3+2]=out.z;
         writevalues[24+threadIdx.x]=ok;
      }

      if(tidx<24)
      {
         float outwrite=writevalues[tidx];
         ok2=writevalues[24+tidx/3];
         if (ok2==1)
         {
            outptr[blockIdx.z*outstr0+(4*blockIdx.y+ty)*outstr1+(blockIdx.x*8)*outstr2+tidx]=outwrite;
         }
      }
   }
   }


}


static int cunxn_Resize_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *tmp = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "tmp", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  float scale = luaT_getfieldchecknumber(L, 1, "scale");

  input = THCudaTensor_newContiguous(input); // should be contiguous already
  
  int bs       = input->size[0];
  int ih       = input->size[1];
  int iw       = input->size[2];
  int nPlanes  = input->size[3];
  assert(nPlanes==3);
  
  cudaError_t result;
  
  int outy=ih*scale;
  int outx=iw*scale;

  THCudaTensor_resize4d(tmp, bs, ih, iw, 4);  
  THCudaTensor_fill(tmp, 0);  
  THCudaTensor_resize4d(output, bs,  outy, outx, 3);  
  THCudaTensor_fill(output, 0);  
  float * inputptr=THCudaTensor_data(input);
  float * tmpptr=THCudaTensor_data(tmp);
  float * outptr=THCudaTensor_data(output);
  
  cudaMemcpy2D(tmpptr, 4*sizeof(float), inputptr, 3*sizeof(float), 3*sizeof(float), bs*ih*iw ,cudaMemcpyDeviceToDevice);

   result = cudaMemcpy2D(tmpptr, 4*sizeof(float), inputptr, 3*sizeof(float), 3*sizeof(float), bs*ih*iw ,cudaMemcpyDeviceToDevice);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy2D -  %s\n", cudaGetErrorString(result));
		return 1;
	}  

  
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
  cudaArray* imgarray;
  cudaExtent ex = make_cudaExtent(iw, ih, bs);
  

   result = cudaMalloc3DArray(&imgarray, &channelDesc, ex, cudaArrayLayered);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMalloc3DArray -  %s\n", cudaGetErrorString(result));
		return 1;
	}  

  cudaMemcpy3DParms myParms = {0};
  memset(&myParms, 0, sizeof(myParms));
  myParms.srcPtr.pitch = sizeof(float) * iw * 4;
  myParms.srcPtr.ptr = tmpptr;
  myParms.srcPtr.xsize = iw;
  myParms.srcPtr.ysize = ih;

  myParms.srcPos.x = 0;
  myParms.srcPos.y = 0;
  myParms.srcPos.z = 0;
  
  myParms.dstArray = imgarray;

  myParms.dstPos.x = 0;
  myParms.dstPos.y = 0;
  myParms.dstPos.z = 0;

  myParms.extent.width = iw;
  myParms.extent.depth = bs;
  myParms.extent.height = ih;

  myParms.kind = cudaMemcpyDeviceToDevice;

  result = cudaMemcpy3D(&myParms);
	if (result != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy3D - failed to copy 1 - %s\n", cudaGetErrorString(result));
		return 1;
	}
	
    

    texRef.addressMode[0]   = cudaAddressModeBorder;
    texRef.addressMode[1]   = cudaAddressModeBorder;
    texRef.filterMode       = cudaFilterModeLinear;
    texRef.normalized       = 1;
	
	 cudaBindTextureToArray(texRef, imgarray);
	
	
	
    int instr0    = input->stride[0];
    int instr1    = input->stride[1];
    int instr2    = input->stride[2];
    int instr3    = input->stride[3];
    int outstr0    = output->stride[0];
    int outstr1    = output->stride[1];
    int outstr2    = output->stride[2];
    int outstr3    = output->stride[3];
    
    dim3 blocks((outx+31)/32, outy, bs);
    dim3 threads(3,32);
    
    //resizeKernel <<<blocks, threads>>>(outptr, outstr0, outstr1, outstr2, outstr3, outx);
   
    dim3 blockstiled((outx+7)/8, (outy+3)/4, bs);
    dim3 threadstiled(8,4);
    
    resizeTiledKernel <<<blockstiled, threadstiled>>>(outptr, outstr0, outstr1, outstr2, outstr3, outx, outy);

	 cudaUnbindTexture(texRef);
	
	
	


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in Resize.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
 
  cudaFreeArray(imgarray);
 
  // final cut:
  THCudaTensor_free(input); 
  //THCudaTensor_free(tmp); 
  //THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}











static int cunxn_Resize_updateGradInput(lua_State *L)
{


  return 1;
}

static const struct luaL_Reg cunxn_Resize__ [] = {
  {"Resize_updateOutput", cunxn_Resize_updateOutput},
  {"Resize_updateGradInput", cunxn_Resize_updateGradInput},
  {NULL, NULL}
};

static void cunxn_Resize_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_Resize__, "nxn");
  lua_pop(L,1);
}
