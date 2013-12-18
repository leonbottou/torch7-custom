#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

#define MIN(a,b) (a) < (b) ? (a) : (b)
#define MAX(a,b) (a) > (b) ? (a) : (b)


__global__ void globalMaxPool(float *ptrinput, float *ptroutput, float *ptrindices, const int isize1, const int isize2, const int nPlanes)
{
	// each thread does a plane
	const int tidx = blockDim.x * blockIdx.x + threadIdx.x;

	int i;

	float out=-2e38;
	int index=-1;

	// nPlanes better be a multiple of 32 for coalesced reads

	if (tidx<nPlanes) {
		for(i=0; i<isize1*isize2; i++) {
			float in = ptrinput[i*nPlanes+tidx];
			if (in>out) {
				out=in;
				index=i;
			}
		}	
	
		ptroutput[tidx]  = out;
		ptrindices[tidx] = index;
	}
}


__global__ void globalMaxPoolBackward(float *ptrgradinput, float *ptrgradoutput, float *ptrindices, const int isize1, const int isize2, const int nPlanes)
{

	// this one can go full-speed : each block does a pixel
	// but nPlanes should be a multiple of 32 if possible for coalesced writes

	const int tidx = threadIdx.x;
	const int blk  = blockDim.x;
	const int pixidx = gridDim.x * blockIdx.y + blockIdx.x;
	const int valuesperthread = (nPlanes + blk - 1) / blk;

	int k;


	// move pointers
	ptrgradinput   += pixidx * nPlanes ;
	for(k=0; k<valuesperthread; k++) {
		if(k*blk + tidx < nPlanes) {
			float index = ptrindices[k*blk + tidx];
			float gradoutvalue = ptrgradoutput[k*blk + tidx];
			float gradinvalue = pixidx==index ? gradoutvalue : 0;
			ptrgradinput[k*blk + tidx] = gradinvalue;
		}
	}	
	

}




static int cunn_SpatialGlobalMaxPoolingNew_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");

  //luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");

  // input should be contiguous already but... well.
  input = THCudaTensor_newContiguous(input);

  // find the size of kernelslices
  long isize1 = input->size[0];
  long isize2 = input->size[1];
  long nPlanes = input->size[2];
//  assert(nPlanes%32 == 0);

  THCudaTensor_resize3d(output, 1, 1, nPlanes);
  THCudaTensor_resizeAs(indices, output);
  

  float* ptroutput  = THCudaTensor_data(output);
  float* ptrinput   = THCudaTensor_data(input);
  float* ptrindices   = THCudaTensor_data(indices);
  

  // cuda blocks & threads:
  dim3 blocks ((nPlanes + 31) / 32);
  dim3 threads (32);

  globalMaxPool <<<blocks, threads>>> (ptrinput, ptroutput, ptrindices, isize1, isize2, nPlanes);



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in globalMaxPool: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }


  // final cut:
  //THCudaTensor_free(input); 
  //THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}





static int cunn_SpatialGlobalMaxPoolingNew_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");

  long isize1 = input->size[0];
  long isize2 = input->size[1];
  long nPlanes = input->size[2];
//  assert(nPlanes%32 == 0);

  long outsize1 = gradOutput->size[0];
  long outsize2 = gradOutput->size[1];

  assert(outsize1 == 1);
  assert(outsize2 == 1);

  THCudaTensor_resizeAs(gradInput, input);

  dim3 blocks (isize1, isize2);
  dim3 threads (32);

  float* ptrindices  = THCudaTensor_data(indices);
  float* ptrgradoutput  = THCudaTensor_data(gradOutput);
  float* ptrgradinput   = THCudaTensor_data(gradInput);


  globalMaxPoolBackward <<<blocks, threads>>> (ptrgradinput, ptrgradoutput, ptrindices, isize1, isize2, nPlanes);
  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in globalMaxPoolBackward: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
  return 1;
}



static const struct luaL_Reg cunn_SpatialGlobalMaxPoolingNew__ [] = {
  {"SpatialGlobalMaxPoolingNew_updateOutput", cunn_SpatialGlobalMaxPoolingNew_updateOutput},
  {"SpatialGlobalMaxPoolingNew_updateGradInput", cunn_SpatialGlobalMaxPoolingNew_updateGradInput},
  {NULL, NULL}
};

static void cunn_SpatialGlobalMaxPoolingNew_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialGlobalMaxPoolingNew__, "nn");
  lua_pop(L,1);
}
