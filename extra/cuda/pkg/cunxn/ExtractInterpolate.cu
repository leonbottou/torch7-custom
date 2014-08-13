#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif


static texture<float4, cudaTextureType2DLayered> texRef2;

__global__ void extractInterpolateKernel(float* outptr, int outstr0, int outstr1, int outstr2, int outstr3, int outx, int outy, float y1, float x1, float y2, float x2, float y3, float x3, float y4, float x4)
{
   // blockIdx.x = 0, iw*scale-1 / 8
   // blockIdx.y = 0, ih*scale-1 / 4
   // blockIdx.z = 0, bs-1
   // threadIdx.x= 0, 7
   // threadIdx.y= 0, 3
   

   // we must compute the pixel position in the 1x1 normalized texture
   // we have to interpolate the coordinates

   // this is the mapping between a given thread and the 0-1 texture space
   float coordx0 = (float)(blockIdx.x*8+threadIdx.x)/outx;
   float coordy0 = (float)(blockIdx.y*4+threadIdx.y)/outy;

   // we put some offset (y_i, x_i) are the input coordinates of the output corners (1 : top-left, 2 : top-right, 3 : bot-right, 4 : bot-left)
   float upinter = (x1+(coordx0*(x2-x1)));
   float downinter = (x4+(coordx0*(x3-x4)));
   float leftinter = (y1+(coordy0*(y4-y1)));
   float rightinter = (y2+(coordy0*(y3-y2)));

   float coordx = upinter + coordy0*(downinter - upinter);
   float coordy = leftinter + coordx0*(rightinter - leftinter);
   
   int tidx = threadIdx.y*blockDim.x+threadIdx.x;
   float4 out;
   float ok=0;
   float ok2;
   __shared__ volatile float writevalues[32];
   if (coordx<1 && coordy<1 && coordx0 <1 && coordy0 <1)
   {
   // read :
      out    = tex2DLayered(texRef2, coordx, coordy, blockIdx.z);
      ok=1;
   }

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

static int cunxn_ExtractInterpolate_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *tmp = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "tmp", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  int outy = luaT_getfieldcheckint(L, 1, "targety");
  int outx = luaT_getfieldcheckint(L, 1, "targetx");
  int y1int = luaT_getfieldcheckint(L, 1, "y1");
  int y2int = luaT_getfieldcheckint(L, 1, "y2");
  int y3int = luaT_getfieldcheckint(L, 1, "y3");
  int y4int = luaT_getfieldcheckint(L, 1, "y4");
  int x1int = luaT_getfieldcheckint(L, 1, "x1");
  int x2int = luaT_getfieldcheckint(L, 1, "x2");
  int x3int = luaT_getfieldcheckint(L, 1, "x3");
  int x4int = luaT_getfieldcheckint(L, 1, "x4");

  input = THCudaTensor_newContiguous(input); // should be contiguous already
  
  int bs       = input->size[0];
  int ih       = input->size[1];
  int iw       = input->size[2];
  int nPlanes  = input->size[3];
  assert(nPlanes==3);
  
  float y1 = ((float)y1int-1)/(float)(ih-1);
  float y2 = ((float)y2int-1)/(float)(ih-1);
  float y3 = ((float)y3int-1)/(float)(ih-1);
  float y4 = ((float)y4int-1)/(float)(ih-1);
  float x1 = ((float)x1int-1)/(float)(iw-1);
  float x2 = ((float)x2int-1)/(float)(iw-1);
  float x3 = ((float)x3int-1)/(float)(iw-1);
  float x4 = ((float)x4int-1)/(float)(iw-1);
  
  
  
  
  cudaError_t result;

  THCudaTensor_resize4d(tmp, bs, ih, iw, 4);  
  THCudaTensor_fill(tmp, 0);  
  THCudaTensor_resize4d(output, bs,  outy, outx, 3);  
  THCudaTensor_fill(output, 0);  
  float * inputptr=THCudaTensor_data(input);
  float * tmpptr=THCudaTensor_data(tmp);
  float * outptr=THCudaTensor_data(output);
  
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
	
    

    texRef2.addressMode[0]   = cudaAddressModeBorder;
    texRef2.addressMode[1]   = cudaAddressModeBorder;
    texRef2.filterMode       = cudaFilterModeLinear;
    texRef2.normalized       = 1;
	
	 cudaBindTextureToArray(texRef2, imgarray);
	
	
	
    int instr0    = input->stride[0];
    int instr1    = input->stride[1];
    int instr2    = input->stride[2];
    int instr3    = input->stride[3];
    int outstr0    = output->stride[0];
    int outstr1    = output->stride[1];
    int outstr2    = output->stride[2];
    int outstr3    = output->stride[3];
    
    dim3 blockstiled((outx+7)/8, (outy+3)/4, bs);
    dim3 threadstiled(8,4);
    
    //printf("%f, %f, %f, %f, %f, %f, %f, %f\n", y1, x1, y2, x2, y3, x3, y4, x4);
    
    extractInterpolateKernel <<<blockstiled, threadstiled>>>(outptr, outstr0, outstr1, outstr2, outstr3, outx, outy, y1, x1, y2, x2, y3, x3, y4, x4);

	 cudaUnbindTexture(texRef2);
	
	
	


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ExtractInterpolate.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
 
  cudaFreeArray(imgarray);
 
  // final cut:
  THCudaTensor_free(input); 
  //THCudaTensor_free(tmp); 
  //THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}











static int cunxn_ExtractInterpolate_updateGradInput(lua_State *L)
{


  return 1;
}

static const struct luaL_Reg cunxn_ExtractInterpolate__ [] = {
  {"ExtractInterpolate_updateOutput", cunxn_ExtractInterpolate_updateOutput},
  {"ExtractInterpolate_updateGradInput", cunxn_ExtractInterpolate_updateGradInput},
  {NULL, NULL}
};

static void cunxn_ExtractInterpolate_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_ExtractInterpolate__, "nxn");
  lua_pop(L,1);
}
