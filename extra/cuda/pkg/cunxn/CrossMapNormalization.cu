#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

/*
 * Description:
 *    this function crossmap-normalizes one single row
 *    along the innermost dimension
 *    Nd input, Nd output
 */


template <int maxnumplanes> __global__ void CrossMapNormalization_output(float *input, float *output, float *z, const int nPlanes, const float k, const float alpha, const float beta, const int n)
{
  // one block = one pixel (x size feature maps)
  const int pixidx = blockIdx.y * gridDim.x + blockIdx.x;
  const int blk=blockDim.x;
  const int tidx = threadIdx.x;
  const int valuesperthread = nPlanes/blockDim.x;
  
  // input offset:
  const int offset = pixidx*nPlanes;

  // move pointers
  input  += offset;
  output += offset;  
  z 	 += offset;

  float alphan = alpha / n;
  int i,j;


  __shared__ float pixvalues[maxnumplanes];

// coalesced read...
  for (i=0; i<valuesperthread; i++) {
	pixvalues[i*blk + tidx] = input[i*blk + tidx];
  }

  for (i=0; i<valuesperthread; i++) {
         float tmpz = 0;
         int startf = i*blk + tidx - n/2;
         int endf = startf + n;

         for(j=startf; j<endf; j++)
           {
		if(j>-1 && j<nPlanes)
                tmpz += pixvalues[j]*pixvalues[j];
           }
	tmpz=pow(k+tmpz*alphan,-beta);
	z[i*blk + tidx]=tmpz;
	output[i*blk + tidx]=tmpz*pixvalues[i*blk + tidx];
  }
}





template <int maxnumplanes> __global__ void CrossMapNormalization_gradInput(float *input, float* gradOutput, float* gradInput, float *z, const int nPlanes, const float k, const float alpha, const float beta, const int n)
{
  // one block = one pixel (x size feature maps)
  const int pixidx = blockIdx.y * gridDim.x + blockIdx.x;
  const int blk	= blockDim.x;
  const int tidx = threadIdx.x;
  const int valuesperthread = nPlanes/blockDim.x;
  
  // input offset:
  const int offset = pixidx*nPlanes;

  // move pointers
  input  	+= offset;
  gradOutput 	+= offset;  
  gradInput 	+= offset;  
  z 	 	+= offset;

  int i,j;

  float alphan = alpha / n;

  __shared__ float pixvalues[maxnumplanes];
  __shared__ float gradoutvalues[maxnumplanes];
  __shared__ float zvalues[maxnumplanes];

// coalesced reads...
  for (i=0; i<valuesperthread; i++) {
	pixvalues[i*blk + tidx] = input[i*blk + tidx];
	gradoutvalues[i*blk + tidx] = gradOutput[i*blk + tidx];
	zvalues[i*blk + tidx] = z[i*blk + tidx];
  }

      for(i = 0; i < valuesperthread; i++)
       {
         float gradi = 0;
         float ai = pixvalues[i*blk + tidx];
         int endo = i*blk + tidx + n/2 + 1;
         int starto = endo - n;
         for (j=starto; j<endo; j++)
           {
		if(j>-1 && j< nPlanes) {
			float aj = pixvalues[j];
			float gj = gradoutvalues[j];
			gradi += (i*blk + tidx == j) ? gj * pow(zvalues[j], -beta) : 0;
			gradi -= gj * 2 * alphan * beta * ai * aj * pow(zvalues[j], -beta-1);
		}
           }
         gradInput[i*blk + tidx]=gradi;
       }

}





static int cunxn_CrossMapNormalization_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *z = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "z", "torch.CudaTensor");
  float alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float k = luaT_getfieldchecknumber(L, 1, "k");
  long n = luaT_getfieldcheckint(L, 1, "n");


  input = THCudaTensor_newContiguous(input); // should be contiguous already
  

  THCudaTensor_resizeAs(output, input);
  THCudaTensor_resizeAs(z, input);

  float *input_data = THCudaTensor_data(input);
  float *output_data = THCudaTensor_data(output);
  float *z_data = THCudaTensor_data(z);

  long isize1 = input->size[0];
  long isize2 = input->size[1];
  long nPlanes = input ->size[2];

  assert(nPlanes < 4097); // number of planes must be at most 4096 (or there will be shared memory issues...)

  // cuda blocks & threads:
  dim3 blocks(isize1, isize2);
  dim3 threads(32);

  // kernel:

	if (nPlanes >3072) {
  		CrossMapNormalization_output <4096> <<<blocks, threads>>> (input_data, output_data, z_data, nPlanes, k, alpha, beta, n);	  
	} else if (nPlanes >2048) {
  		CrossMapNormalization_output <3072> <<<blocks, threads>>> (input_data, output_data, z_data, nPlanes, k, alpha, beta, n);	  
	} else if (nPlanes >1536) {
  		CrossMapNormalization_output <2048> <<<blocks, threads>>> (input_data, output_data, z_data, nPlanes, k, alpha, beta, n);	  
	} else if (nPlanes >1024) {
  		CrossMapNormalization_output <1536> <<<blocks, threads>>> (input_data, output_data, z_data, nPlanes, k, alpha, beta, n);	  
	} else if (nPlanes >768) {
  		CrossMapNormalization_output <1024> <<<blocks, threads>>> (input_data, output_data, z_data, nPlanes, k, alpha, beta, n);	  
	} else if (nPlanes >512) {
  		CrossMapNormalization_output <768> <<<blocks, threads>>> (input_data, output_data, z_data, nPlanes, k, alpha, beta, n);	  
	} else if (nPlanes >384) {
  		CrossMapNormalization_output <512> <<<blocks, threads>>> (input_data, output_data, z_data, nPlanes, k, alpha, beta, n);	  
	} else if (nPlanes >256) {
  		CrossMapNormalization_output <384> <<<blocks, threads>>> (input_data, output_data, z_data, nPlanes, k, alpha, beta, n);	  
	} else if (nPlanes >128) {
  		CrossMapNormalization_output <256> <<<blocks, threads>>> (input_data, output_data, z_data, nPlanes, k, alpha, beta, n);	  
	} else {
  		CrossMapNormalization_output <128> <<<blocks, threads>>> (input_data, output_data, z_data, nPlanes, k, alpha, beta, n);	  
	}


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in CrossMapNormalization.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }
 
  // final cut:
  THCudaTensor_free(input); 
  //THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}











static int cunxn_CrossMapNormalization_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *z = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "z", "torch.CudaTensor");
//  int dimension  = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *gradInput  = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  float alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float k = luaT_getfieldchecknumber(L, 1, "k");
  long n = luaT_getfieldcheckint(L, 1, "n");

  THCudaTensor_resizeAs(gradInput, input);
  THCudaTensor_zero(gradInput);
  
  float *input_data = THCudaTensor_data(input);
  float *gradInput_data = THCudaTensor_data(gradInput);
  float *gradOutput_data = THCudaTensor_data(gradOutput);
  float *z_data = THCudaTensor_data(z);


  long isize1 = input->size[0];
  long isize2 = input->size[1];
  long nPlanes = input->size[2];

  assert(nPlanes < 4097); // number of planes must be at most 4096 (or there will be shared memory issues...)

  // cuda blocks & threads:
  dim3 blocks(isize1, isize2);
  dim3 threads(32);
  
  // kernel:

	if (nPlanes >3072) {
		CrossMapNormalization_gradInput <4096> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, nPlanes, k, alpha, beta, n);
	} else if (nPlanes >2048) {
		CrossMapNormalization_gradInput <3072> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, nPlanes, k, alpha, beta, n);
	} else if (nPlanes >1536) {
		CrossMapNormalization_gradInput <2048> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, nPlanes, k, alpha, beta, n);
	} else if (nPlanes >1024) {
		CrossMapNormalization_gradInput <1536> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, nPlanes, k, alpha, beta, n);
	} else if (nPlanes >768) {
		CrossMapNormalization_gradInput <1024> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, nPlanes, k, alpha, beta, n);
	} else if (nPlanes >512) {
		CrossMapNormalization_gradInput <768> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, nPlanes, k, alpha, beta, n);
	} else if (nPlanes >384) {
		CrossMapNormalization_gradInput <512> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, nPlanes, k, alpha, beta, n);
	} else if (nPlanes >256) {
		CrossMapNormalization_gradInput <384> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, nPlanes, k, alpha, beta, n);
	} else if (nPlanes >128) {
		CrossMapNormalization_gradInput <256> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, nPlanes, k, alpha, beta, n);
	} else {
		CrossMapNormalization_gradInput <128> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, nPlanes, k, alpha, beta, n);
	}


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in CrossMapNormalization.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  return 1;
}

static const struct luaL_Reg cunxn_CrossMapNormalization__ [] = {
  {"CrossMapNormalization_updateOutput", cunxn_CrossMapNormalization_updateOutput},
  {"CrossMapNormalization_updateGradInput", cunxn_CrossMapNormalization_updateGradInput},
  {NULL, NULL}
};

static void cunxn_CrossMapNormalization_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_CrossMapNormalization__, "nxn");
  lua_pop(L,1);
}
