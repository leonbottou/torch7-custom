#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

#define MIN(a,b) (a) < (b) ? (a) : (b)
#define MAX(a,b) (a) > (b) ? (a) : (b)


__global__ void maxPool(float *ptrinput, float *ptroutput, const int isize1, const int isize2, const int outsize1, const int outsize2, const int nOutputPlane, const int poolH, const int poolW, const int pooldH, const int pooldW, const int valuesperthread)
{
	const int tidx = threadIdx.x;
	const int blk  = blockDim.x;
	const int pixi = blockIdx.x;
	const int pixj = blockIdx.y;

	int i,j,k;

	// move pointers
	ptrinput   += (pixi * pooldH * isize2 + pixj * pooldW) * nOutputPlane ;
	ptroutput  += (pixi * outsize2 + pixj) * nOutputPlane ;
	const int stridej = nOutputPlane;
	const int stridei = (isize2 - poolW) * nOutputPlane;
	const int stridek = - poolH * isize2 * nOutputPlane;

	for(k=0; k<valuesperthread; k++) {
		float out=-2e38; 
		for(i=0; i<poolH; i++) {
			for(j=0; j<poolW; j++) {
				out=MAX(out, ptrinput[k*blk+tidx]);
				ptrinput += stridej;
			}
			ptrinput += stridei;
		}
		ptroutput[k*blk+tidx]=out;
		ptrinput +=stridek;
	}	

}




static int cunn_SpatialMaxPoolingNew_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  long poolW = luaT_getfieldcheckint(L, 1, "poolW");
  long poolH = luaT_getfieldcheckint(L, 1, "poolH");
  long dW = luaT_getfieldcheckint(L, 1, "dW");
  long dH = luaT_getfieldcheckint(L, 1, "dH");


//  long padup = luaT_getfieldcheckint(L, 1, "padup");
//  long paddown = luaT_getfieldcheckint(L, 1, "paddown");
//  long padleft = luaT_getfieldcheckint(L, 1, "padleft");
//  long padright = luaT_getfieldcheckint(L, 1, "padright");

  //luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");


  // input should be contiguous already but... well.
  input = THCudaTensor_newContiguous(input);

  // find the size of kernelslices
  long isize1 = input->size[0];
  long isize2 = input->size[1];
  long isize3 = input->size[2];
  assert(isize3%32 == 0);

  long outsize1 = (isize1 - poolH) / dH + 1;
  long outsize2 = (isize2 - poolW) / dW + 1;

  THCudaTensor_resize3d(output, outsize1, outsize2, isize3);

  float* ptroutput  = THCudaTensor_data(output);
  float* ptrinput   = THCudaTensor_data(input);


  // cuda blocks & threads:
  dim3 blocks (outsize1, outsize2);
  dim3 threads (32);
  long valuesperthread=isize3/32;

  maxPool<<<blocks,threads>>>(ptrinput, ptroutput, isize1, isize2, outsize1, outsize2, isize3, poolH, poolW, dH, dW, valuesperthread);



  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in copyPixelsInSlices: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }


  // final cut:
  THCudaTensor_free(input); 
  //THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}





static int cunn_SpatialMaxPoolingNew_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  luaL_argcheck(L, dW == 1, 1, "dW must be 1 (this is only a limit for CudaTensors)");
  luaL_argcheck(L, dH == 1, 1, "dH must be 1 (this is only a limit for CudaTensors)");

  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  if (input->nDimension == 3)
  {
    /* check dims */
    THArgCheck(nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane");

    /* gradient to input */
    THCudaTensor *tweight = THCudaTensor_newTranspose(weight,0,1);
    THCudaTensor_conv2Dmv(gradInput, 0.0, gradOutput, tweight, dH, dW, "fc");
    THCudaTensor_free(tweight);
  }
  else 
  {
    /* check dims */
    THArgCheck(nOutputPlane == gradOutput->size[1], 1, "Number of output features is not equal to nOutputPlane");

    /* gradient to input */
    THCudaTensor *tweight = THCudaTensor_newTranspose(weight,0,1);
    THCudaTensor_conv2Dmm(gradInput, 0.0, gradOutput, tweight, dH, dW, "fc");
    THCudaTensor_free(tweight);    
  }

  return 1;
}



static const struct luaL_Reg cunn_SpatialMaxPoolingNew__ [] = {
  {"SpatialMaxPoolingNew_updateOutput", cunn_SpatialMaxPoolingNew_updateOutput},
  {"SpatialMaxPoolingNew_updateGradInput", cunn_SpatialMaxPoolingNew_updateGradInput},
  {NULL, NULL}
};

static void cunn_SpatialMaxPoolingNew_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialMaxPoolingNew__, "nn");
  lua_pop(L,1);
}
