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
	// each thread does a pixel of the output
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
//	const int stridek = (isize1 - poolH) * isize2 * nOutputPlane;
	float * ptrinputsave = ptrinput;

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
		ptrinput =ptrinputsave;
	}	

}


__global__ void maxPoolBackward(float *ptrinput, float *ptroutput, float *ptrgradinput, float *ptrgradoutput, const int isize1, const int isize2, const int outsize1, const int outsize2, const int nOutputPlane, const int poolH, const int poolW, const int pooldH, const int pooldW, const int valuesperthread)
{

	// this one is a bit tricky : we have to add up the gradient if the pooling overlaps...
	// so each block (each thread ?) will do one pixel of the input...
	// 1) find which outputs are related to the input
	// 2) go

	const int tidx = threadIdx.x;
	const int blk  = blockDim.x;
	const int pixi = blockIdx.x;
	const int pixj = blockIdx.y;

        const int imin=(pixi - (poolH - 1) + (pooldH -1))/pooldH > 0 ? (pixi - (poolH - 1) + (pooldH -1))/pooldH : 0 ;
        const int jmin=(pixj - (poolW - 1) + (pooldW -1))/pooldW > 0 ? (pixj - (poolW - 1) + (pooldW -1))/pooldW : 0 ;
        const int imax= pixi / pooldH < outsize1 ? pixi / pooldH : outsize1 - 1 ;
        const int jmax= pixj / pooldW < outsize2 ? pixj / pooldW : outsize2 - 1 ;

	int i,j,k;

	// move pointers
	ptrinput   += (pixi * isize2 + pixj) * nOutputPlane ;
	ptrgradinput   += (pixi * isize2 + pixj) * nOutputPlane ;
	ptroutput  += (imin * outsize2 + jmin) * nOutputPlane ;
	ptrgradoutput  += (imin * outsize2 + jmin) * nOutputPlane ;
	float * ptroutputsave = ptroutput;
	float * ptrgradoutputsave = ptrgradoutput;
	
	const int stridej = nOutputPlane;
	const int stridei = (outsize2 -jmax+jmin-1) * nOutputPlane;
//	const int stridek = (imax+imin-1 ) * outsize2 * nOutputPlane; // this one just brings the pointer back to where it was...

	for(k=0; k<valuesperthread; k++) {
		float pixvalue=ptrinput[k*blk+tidx];
//		float gradinputvalue=0;
		for(i=imin; i<imax+1; i++) {
			for(j=jmin; j<jmax+1; j++) {
				float out=ptroutput[k*blk+tidx];
				if(pixvalue==out) {
					ptrgradinput[k*blk+tidx] += ptrgradoutput[k*blk+tidx];
//					gradinputvalue += ptrgradoutput[k*blk+tidx];
				}
				ptroutput += stridej;
				ptrgradoutput += stridej;
			}
			ptroutput += stridei;
			ptrgradoutput += stridei;
		}
//		ptrgradinput[k*blk+tidx]=gradinputvalue;
		ptroutput = ptroutputsave;
		ptrgradoutput = ptrgradoutputsave;
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
    printf("error in maxPool: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }


  // final cut:
  //THCudaTensor_free(input); 
  //THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}





static int cunn_SpatialMaxPoolingNew_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  long poolW = luaT_getfieldcheckint(L, 1, "poolW");
  long poolH = luaT_getfieldcheckint(L, 1, "poolH");


  long isize1 = input->size[0];
  long isize2 = input->size[1];
  long isize3 = input->size[2];
  assert(isize3%32 == 0);

  long outsize1 = output->size[0];
  long outsize2 = output->size[1];

  THCudaTensor_resizeAs(gradInput, input);

  dim3 blocks (isize1, isize2);
  dim3 threads (32);
  long valuesperthread=isize3/32;

  float* ptroutput  = THCudaTensor_data(output);
  float* ptrinput   = THCudaTensor_data(input);
  float* ptrgradoutput  = THCudaTensor_data(gradOutput);
  float* ptrgradinput   = THCudaTensor_data(gradInput);


  maxPoolBackward <<<blocks,threads>>>(ptrinput, ptroutput, ptrgradinput, ptrgradoutput, isize1, isize2, outsize1, outsize2, isize3,  poolH, poolW, dH, dW, valuesperthread);
  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in maxPoolBackward: %s\n", cudaGetErrorString(err));
    THError("aborting");
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
