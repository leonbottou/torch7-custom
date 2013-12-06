#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif


__global__ void copyPixelsInSlices(float *ptrinput, float *ptrkslices,
	int dH, int dW, int kH, int kW, int size1, int size2, int isize1, int isize2, int nInputPlane, int valuesperthread)
{
	const int pixi=blockIdx.x;
	const int pixj=blockIdx.y;
	const int blk =blockDim.x;
	const int tidx=threadIdx.x;

        int imin=(pixi - (kH - 1) + (dH -1))/dH > 0 ? (pixi - (kH - 1) + (dH -1))/dH : 0 ;
        int jmin=(pixj - (kW - 1) + (dW -1))/dW > 0 ? (pixj - (kW - 1) + (dW -1))/dW : 0 ;
        int imax= pixi / dH < size1 ? pixi / dH : size1 - 1 ;
        int jmax= pixj / dW < size2 ? pixj / dW : size2 - 1 ;

	int i;
	int j;
	int k;

	ptrinput   += (pixi * isize2 + pixj) * nInputPlane ;
	ptrkslices += ((imin * size2  + jmin) * kH * kW +  (pixi - imin * dH) * kW + (pixj - jmin*dW) ) * nInputPlane;

	int stridej = (kH*kW - dW) * nInputPlane;
	int stridei = (((size2-jmax+jmin-1)*kH -dH)*kW  + (jmax-jmin+1)*dW)*nInputPlane;
	
	for(i=imin; i<imax+1; i++) {
		for(j=jmin; j<jmax+1; j++) {
			for(k=0; k<valuesperthread; k++) {
				ptrkslices[k*blk+tidx]=ptrinput[k*blk+tidx];
			}
			ptrkslices += stridej;
		}
		ptrkslices += stridei;
	}	
}


template <int maxnumplanes> __global__ void copyPixelsInSlicesSharedMem(float *ptrinput, float *ptrkslices,
	int dH, int dW, int kH, int kW, int size1, int size2, int isize1, int isize2, int nInputPlane, int valuesperthread)
{
	const int pixi=blockIdx.x;
	const int pixj=blockIdx.y;
	const int blk =blockDim.x;
	const int tidx=threadIdx.x;

	__shared__ float pixvalues[maxnumplanes];

        int imin=(pixi - (kH - 1) + (dH -1))/dH > 0 ? (pixi - (kH - 1) + (dH -1))/dH : 0 ;
        int jmin=(pixj - (kW - 1) + (dW -1))/dW > 0 ? (pixj - (kW - 1) + (dW -1))/dW : 0 ;
        int imax= pixi / dH < size1 ? pixi / dH : size1 - 1 ;
        int jmax= pixj / dW < size2 ? pixj / dW : size2 - 1 ;

	int i;
	int j;
	int k;

	ptrinput   += (pixi * isize2 + pixj) * nInputPlane ;
	ptrkslices += ((imin * size2  + jmin) * kH * kW +  (pixi - imin * dH) * kW + (pixj - jmin*dW) ) * nInputPlane;

	int stridej = (kH*kW - dW) * nInputPlane;
	int stridei = (((size2-jmax+jmin-1)*kH -dH)*kW  + (jmax-jmin+1)*dW)*nInputPlane;


	// load the stuff in shared memory first...
	for(k=0; k<valuesperthread; k++) {
		pixvalues[k*blk+tidx]=ptrinput[k*blk+tidx];
	}
	

	
	for(i=imin; i<imax+1; i++) {
		for(j=jmin; j<jmax+1; j++) {
			for(k=0; k<valuesperthread; k++) {
				ptrkslices[k*blk+tidx]=pixvalues[k*blk+tidx];
			}
			ptrkslices += stridej;
		}
		ptrkslices += stridei;
	}	
}


template <int maxnumplanes> __global__ void copyBiasToOutputsSharedMem(float *ptrbias, float *ptroutput, int size1, int size2, int nOutputPlane, int valuesperthread)
{
	// each thread has a value to manage...
	const int blk =blockDim.x;
	const int tidx=threadIdx.x;

	__shared__ float biasvalues[maxnumplanes];
 
	int i;
	int j;
	int k;

	// load all in shared mem
	for(k=0; k<valuesperthread;k++) {
		biasvalues[k*blk+tidx]=ptrbias[k*blk+tidx];
	}

	// copy to output
	for(i=0; i<size1; i++) {
		for(j=0; j<size2; j++) {
			for(k=0; k<valuesperthread;k++) {
				ptroutput[k*blk+tidx]=biasvalues[k*blk+tidx];
			}
			ptroutput+=nOutputPlane;
		}
	}
}


__global__ void copyBiasToOutputs(float *ptrbias, float *ptroutput, int size1, int size2, int nOutputPlane, int valuesperthread)
{
	// each thread has a value to manage...
	const int blk =blockDim.x;
	const int tidx=threadIdx.x;

	int i;
	int j;
	int k;

	for(i=0; i<size1; i++) {
		for(j=0; j<size2; j++) {
			for(k=0; k<valuesperthread; k++) {
				ptroutput[k*blk+tidx]=ptrbias[k*blk+tidx];
			}
			ptroutput+=nOutputPlane;
		}
	}
}




static int cunn_SpatialConvolutionNew_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *kernels = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
  THCudaTensor *kslicestest = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "kslicestest", "torch.CudaTensor");
  long kW = luaT_getfieldcheckint(L, 1, "kW");
  long kH = luaT_getfieldcheckint(L, 1, "kH");
  long dW = luaT_getfieldcheckint(L, 1, "dW");
  long dH = luaT_getfieldcheckint(L, 1, "dH");
  long shdmem = luaT_getfieldcheckint(L, 1, "shdmem");
  long nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  long nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  //luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");

  assert(nInputPlane%32 == 0);
  assert(nOutputPlane%32 == 0);

  // unfold the input tensor 
  // input should be contiguous already but... well.
  input = THCudaTensor_newContiguous(input);

  // find the size of kernelslices
  long isize1 = input->size[0];
  long isize2 = input->size[1];
  long size1 = (isize1 - kH) / dH + 1;
  long size2 = (isize2 - kW) / dW + 1;

  THCudaTensor* kernelSlices = THCudaTensor_newWithSize1d(size1*size2*kW*kH*nInputPlane);
  THCudaTensor_resize2d(output, size1* size2, nOutputPlane);

  float* ptrkslices = THCudaTensor_data(kernelSlices);
  float* ptroutput  = THCudaTensor_data(output);
  float* ptrinput   = THCudaTensor_data(input);
  float* ptrbias    = THCudaTensor_data(bias);

  // cuda blocks & threads:
  dim3 blocks (isize1, isize2);
  dim3 threads (32);
  long valuesperthread=nInputPlane/32;

  
  //kernel unfold inputs
  if (nInputPlane >1024 || shdmem==0) {
  copyPixelsInSlices<<<blocks, threads>>>(ptrinput, ptrkslices,
	dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, valuesperthread);
  }
  else if (nInputPlane >512) {
        printf("using shared memory 1024 floats\n");
        copyPixelsInSlicesSharedMem <1024> <<<blocks, threads>>>(ptrinput, ptrkslices,
	dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, valuesperthread);
  }
  else if (nInputPlane >384) {
        printf("using shared memory 512 floats\n");
        copyPixelsInSlicesSharedMem <512> <<<blocks, threads>>>(ptrinput, ptrkslices,
	dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, valuesperthread);
  }
  else if (nInputPlane >256) {
        printf("using shared memory 384 floats\n");
        copyPixelsInSlicesSharedMem <384> <<<blocks, threads>>>(ptrinput, ptrkslices,
	dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, valuesperthread);
  }
  else if (nInputPlane >128) {
        printf("using shared memory 256 floats\n");
        copyPixelsInSlicesSharedMem <256> <<<blocks, threads>>>(ptrinput, ptrkslices,
	dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, valuesperthread);
  }
  else {
        printf("using shared memory 128 floats\n");
        copyPixelsInSlicesSharedMem <128> <<<blocks, threads>>>(ptrinput, ptrkslices,
	dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, valuesperthread);
  }
  


  // fill output with biases : use shared memory 48K if possible, there is only one block anyway
  dim3 blocksbias (1);
  dim3 threadsbias (32);

  if (nOutputPlane >12256 || shdmem==0) {
  	copyBiasToOutputs<<<blocksbias, threadsbias>>>(ptrbias, ptroutput, size1, size2, nOutputPlane, nOutputPlane/32);  }
  else {
        printf("using shared memory 12k floats\n");
  	copyBiasToOutputsSharedMem<12256><<<blocksbias, threadsbias>>>(ptrbias, ptroutput, size1, size2, nOutputPlane, nOutputPlane/32);  }


  // unfold conv kernels by resizing
  THCudaTensor_resize2d(kernels, nOutputPlane, kW*kH*nInputPlane);
  THCudaTensor_transpose(kernels, NULL, 0, 1);
  // put kernelslices in matrix mode
  THCudaTensor_resize2d(kernelSlices, size1*size2,kW*kH*nInputPlane);


  // do addmm on output
  
  THCudaTensor_addmm(output, 1,1, kernelSlices, kernels);


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in copyPixelsInSlices: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  THCudaTensor_resize3d(output, size1, size2, nOutputPlane);
 
  THCudaTensor_resizeAs(kslicestest, kernelSlices);
  THCudaTensor_copy(kslicestest, kernelSlices);

  // final cut:
  THCudaTensor_free(input); 
  THCudaTensor_free(kernelSlices); 
  //THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}





static int cunn_SpatialConvolutionNew_updateGradInput(lua_State *L)
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

__global__ void compute_gradBias1(float *gradBias, float *gradOutput, float scale,
                                 int output_n, int output_h, int output_w)
{
  // each block does a plane
  int k = blockIdx.x;
  float *gradOutput_k = gradOutput + (k + threadIdx.y*output_n)*output_h*output_w;

  // offsets
  int i_start = threadIdx.x;
  int i_end = output_w*output_h;
  int i_step = blockDim.x;

  int tid = threadIdx.x + threadIdx.y * blockDim.x;
  int nthreads = blockDim.x * blockDim.y;

  // sum output plane k into partial sum array
  __shared__ float sums[512];
  sums[tid] = 0;
  for (int i=i_start; i<i_end; i+=i_step) {
    sums[tid] += gradOutput_k[i];
  }
  __syncthreads();

  // reduce
  if (tid == 0) {
    for (int i=0; i<nthreads; i++)
      gradBias[k] += scale*sums[i];
  }
}

static int cunn_SpatialConvolutionNew_accGradParameters(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  float scale = luaL_optnumber(L, 4, 1);

  luaL_argcheck(L, dW == 1, 1, "dW must be 1 (this will be fixed soon)");
  luaL_argcheck(L, dH == 1, 1, "dH must be 1 (this will be fixed soon)");

  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");

  float *gradBias_data = THCudaTensor_data(gradBias);
  float *gradOutput_data = THCudaTensor_data(gradOutput);

  if (input->nDimension == 3)
  {
    /* check dims */
    THArgCheck(nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane");

    /* gradient to bias */
    dim3 blocks(nOutputPlane);
    dim3 threads(32);
    compute_gradBias <<<blocks, threads>>> (gradBias_data, gradOutput_data, scale,
                                            gradOutput->size[0], gradOutput->size[1], gradOutput->size[2]);

    /* gradient to kernels */
    THCudaTensor_conv2DRevger(gradWeight, 1.0, scale, input, gradOutput, dH, dW);
  }
  else
  {
    /* check dims */
    THArgCheck(nOutputPlane == gradOutput->size[1], 1, "Number of output features is not equal to nOutputPlane");

    /* gradient to bias */
    dim3 blocks(nOutputPlane);
    long sl;
    for (sl=0; sl<gradOutput->size[0]; sl+=16) {
      int cst = 16;
      if ((cst+sl) > gradOutput->size[0]) cst = gradOutput->size[0] - sl;
      dim3 threads(16, cst);
      compute_gradBias <<<blocks, threads>>> (gradBias_data, gradOutput_data + sl*gradOutput->stride[0], scale,
                                              gradOutput->size[1], gradOutput->size[2], gradOutput->size[3]);
    }

    /* gradient to kernels */
    THCudaTensor_conv2DRevgerm(gradWeight, 1.0, scale, input, gradOutput, dH, dW);
  }

  return 0;
}

static const struct luaL_Reg cunn_SpatialConvolutionNew__ [] = {
  {"SpatialConvolutionNew_updateOutput", cunn_SpatialConvolutionNew_updateOutput},
  {"SpatialConvolutionNew_updateGradInput", cunn_SpatialConvolutionNew_updateGradInput},
  {"SpatialConvolutionNew_accGradParameters", cunn_SpatialConvolutionNew_accGradParameters},
  {NULL, NULL}
};

static void cunn_SpatialConvolutionNew_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_SpatialConvolutionNew__, "nn");
  lua_pop(L,1);
}
