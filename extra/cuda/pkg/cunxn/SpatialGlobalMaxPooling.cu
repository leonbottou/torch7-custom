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
	
	ptrinput    += blockIdx.y * isize1*isize2*nPlanes;
	ptroutput   += blockIdx.y * nPlanes;
	ptrindices  += blockIdx.y * nPlanes;

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

	const int pixidx = gridDim.x * blockIdx.y + blockIdx.x;

	int k;

	// move pointers
	ptrgradinput   += pixidx * nPlanes + blockIdx.z*isize1*isize2*nPlanes ;
	ptrgradoutput  += blockIdx.z*nPlanes ;
	ptrindices     += blockIdx.z*nPlanes ;

	for(k=threadIdx.x; k<nPlanes; k+=blockDim.x) {
			float index = ptrindices[k];
			float gradoutvalue = ptrgradoutput[k];
			float gradinvalue = pixidx==index ? gradoutvalue : 0;
			ptrgradinput[k] = gradinvalue;

	}	
	
/*	for(k=0; k<valuesperthread; k++) {
		if(k*blk + tidx < nPlanes) {
			float index = ptrindices[k*blk + tidx];
			float gradoutvalue = ptrgradoutput[k*blk + tidx];
			float gradinvalue = pixidx==index ? gradoutvalue : 0;
			ptrgradinput[k*blk + tidx] = gradinvalue;
		}
	}	
	
*/
}




static int cunxn_SpatialGlobalMaxPooling_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");

  //luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");

  // input should be contiguous already but... well.
  input = THCudaTensor_newContiguous(input);

  // find the size of kernelslices
  long bs     = input->size[0];
  long isize1 = input->size[1];
  long isize2 = input->size[2];
  long nPlanes = input->size[3];
//  assert(nPlanes%32 == 0);

  THCudaTensor_resize4d(output, bs, 1, 1, nPlanes);
  THCudaTensor_resizeAs(indices, output);
  

  float* ptroutput  = THCudaTensor_data(output);
  float* ptrinput   = THCudaTensor_data(input);
  float* ptrindices   = THCudaTensor_data(indices);
  

  // cuda blocks & threads:
  dim3 blocks ((nPlanes + 31) / 32, bs);
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





static int cunxn_SpatialGlobalMaxPooling_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *indices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "indices", "torch.CudaTensor");

  long bs     = input->size[0];
  long isize1 = input->size[1];
  long isize2 = input->size[2];
  long nPlanes = input->size[3];

  assert(gradOutput->size[1] == 1);
  assert(gradOutput->size[2] == 1);

  THCudaTensor_resizeAs(gradInput, input);

  dim3 blocks (isize1, isize2, bs);
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



static const struct luaL_Reg cunxn_SpatialGlobalMaxPooling__ [] = {
  {"SpatialGlobalMaxPooling_updateOutput", cunxn_SpatialGlobalMaxPooling_updateOutput},
  {"SpatialGlobalMaxPooling_updateGradInput", cunxn_SpatialGlobalMaxPooling_updateGradInput},
  {NULL, NULL}
};

static void cunxn_SpatialGlobalMaxPooling_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_SpatialGlobalMaxPooling__, "nxn");
  lua_pop(L,1);
}
