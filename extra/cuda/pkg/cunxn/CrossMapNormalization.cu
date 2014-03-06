#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

/*
 * This could be improved by using registers + __shfl
 */



template <int maxnumplanes> __global__ void CrossMapNormalization_output(float *input, float *output, float *zsave, int iw, int nplanes, const float k, const float alpha, const float beta, const int n, int str0, int str1)
{
   /* blockIdx.z  = [ 0, bs    ] ()
      blockIdx.y  = [ 0, ih    ] ()
      threadIdx.x = [ 0, 31    ] ()
   */

   float alphan=alpha/n;
   
   __shared__ float pixvalues[maxnumplanes];
   
   input  += blockIdx.z*str0 + blockIdx.y*str1;
   output += blockIdx.z*str0 + blockIdx.y*str1;
   zsave  += blockIdx.z*str0 + blockIdx.y*str1;
   

   // for each pixel 0 -> iw
   for (int pixw=0; pixw<iw; pixw++)
   {
      // load pixels in shared memory
      for (int i=threadIdx.x; i<nplanes;  i+=blockDim.x)
      {
         pixvalues[i]=input[pixw*nplanes+i];   
      }
      
      for (int i=threadIdx.x; i<nplanes;  i+=blockDim.x)
      {
         float z=0;
         int startf = i - n/2;
         int endf   = startf + n;
         startf = (startf < 0) ? 0 : startf;
         endf = (endf > nplanes) ? nplanes : endf;
         
         for (int j=startf; j<endf; j++)
         {
            if(j>-1 && j<nplanes)
            {
               z += pixvalues[j]*pixvalues[j];
            }
         }
         z=k+z*alphan;
         zsave[pixw*nplanes+i]=z;
         output[pixw*nplanes+i]=pixvalues[i]*pow(z,-beta);
      }
   }
}


template <int maxnumplanes> __global__ void CrossMapNormalization_gradInput(float *input, float* gradOutput, float* gradInput, float *zsave, int iw, int nplanes, const float k, const float alpha, const float beta, const int n, int str0, int str1)
{
   /* blockIdx.z  = [ 0, bs    ] ()
      blockIdx.y  = [ 0, ih    ] ()
      threadIdx.x = [ 0, 31    ] ()
   */

   float alphan=alpha/n;

   __shared__ float pixvalues[maxnumplanes];
   __shared__ float gradoutvalues[maxnumplanes];
   __shared__ float zvalues[maxnumplanes];
   
   input  += blockIdx.z*str0 + blockIdx.y*str1;
   gradOutput += blockIdx.z*str0 + blockIdx.y*str1;
   gradInput += blockIdx.z*str0 + blockIdx.y*str1;
   zsave  += blockIdx.z*str0 + blockIdx.y*str1;
   
   // for each pixel 0 -> iw
   for (int pixw=0; pixw<iw; pixw++)
   {
      // load pixels in shared memory
      for (int i=threadIdx.x; i<nplanes;  i+=blockDim.x)
      {
         pixvalues[i]=input[pixw*nplanes+i];
         gradoutvalues[i]=gradOutput[pixw*nplanes+i];   
         zvalues[i]=zsave[pixw*nplanes+i];   
      }
      
      for (int i=threadIdx.x; i<nplanes;  i+=blockDim.x)
      {
         float gradi=0;
         float ai = pixvalues[i];
         int endo = i + n/2 + 1;
         int starto = endo - n;

         for (int j=starto; j<endo; j++)
         {
            if(j>-1 && j<nplanes)
            {
      			float aj = pixvalues[j];
      			float gj = gradoutvalues[j];
      			gradi += (i == j) ? gj * pow(zvalues[j], -beta) : 0;
      			gradi -= gj * 2 * alphan * beta * ai * aj * pow(zvalues[j], -beta-1);
            }
         }
         gradInput[pixw*nplanes+i]=gradi;
      }
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
  int n = luaT_getfieldcheckint(L, 1, "n");


  input = THCudaTensor_newContiguous(input); // should be contiguous already
  

  THCudaTensor_resizeAs(output, input);
  THCudaTensor_resizeAs(z, input);

  float *input_data = THCudaTensor_data(input);
  float *output_data = THCudaTensor_data(output);
  float *z_data = THCudaTensor_data(z);

  int bs       = input->size[0];
  int ih       = input->size[1];
  int iw       = input->size[2];
  int nPlanes  = input->size[3];

  int str0     = input->stride[0];
  int str1     = input->stride[1];

  assert(nPlanes < 4097); // number of planes must be at most 4096 (or there will be shared memory issues...)

  // cuda blocks & threads:
  dim3 blocks(1, ih, bs);
  dim3 threads(32);

  // kernel:
   /*
   template <int maxnumplanes> __global__ void CrossMapNormalization_output(float *input, float *output, float *zsave, int iw, int nplanes, const float k, const float alpha, const float beta, const int n, int str0, int str1)
         blockIdx.z  = [ 0, bs    ] ()
         blockIdx.y  = [ 0, ih    ] ()
         threadIdx.x = [ 0, 31    ] ()
   */   
   
	if (nPlanes >3072) {
  		CrossMapNormalization_output <4096> <<<blocks, threads>>> (input_data, output_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);	  
	} else if (nPlanes >2048) {
	   CrossMapNormalization_output <3072> <<<blocks, threads>>> (input_data, output_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);	  
	} else if (nPlanes >1536) {
	   CrossMapNormalization_output <2048> <<<blocks, threads>>> (input_data, output_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);	  
	} else if (nPlanes >1024) {
	   CrossMapNormalization_output <1536> <<<blocks, threads>>> (input_data, output_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);	  
	} else if (nPlanes >768) {
	   CrossMapNormalization_output <1024> <<<blocks, threads>>> (input_data, output_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);	  
	} else if (nPlanes >512) {
	   CrossMapNormalization_output <768> <<<blocks, threads>>> (input_data, output_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);	  
	} else if (nPlanes >384) {
	   CrossMapNormalization_output <512> <<<blocks, threads>>> (input_data, output_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);	  
	} else if (nPlanes >256) {
	   CrossMapNormalization_output <384> <<<blocks, threads>>> (input_data, output_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);	  
	} else if (nPlanes >128) {
  		CrossMapNormalization_output <256> <<<blocks, threads>>> (input_data, output_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);	  
	} else {
  		CrossMapNormalization_output <128> <<<blocks, threads>>> (input_data, output_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);	  
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


  int bs       = input->size[0];
  int ih       = input->size[1];
  int iw       = input->size[2];
  int nPlanes  = input->size[3];

  int str0     = input->stride[0];
  int str1     = input->stride[1];

  assert(nPlanes < 4097); // number of planes must be at most 4096 (or there will be shared memory issues...)

  // cuda blocks & threads:
  dim3 blocks(1, ih, bs);
  dim3 threads(32);
  
  // kernel:
/*
template <int maxnumplanes> __global__ void CrossMapNormalization_gradInput(float *input, float* gradOutput, float* gradInput, float *zsave, int iw, int nPlanes, const float k, const float alpha, const float beta, const int n, int str0, int str1)
      blockIdx.z  = [ 0, bs    ] ()
      blockIdx.y  = [ 0, ih    ] ()
      threadIdx.x = [ 0, 31    ] ()
   */

	if (nPlanes >3072) {
		CrossMapNormalization_gradInput <4096> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);
	} else if (nPlanes >2048) {
		CrossMapNormalization_gradInput <3072> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);
	} else if (nPlanes >1536) {
		CrossMapNormalization_gradInput <2048> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);
	} else if (nPlanes >1024) {
		CrossMapNormalization_gradInput <1536> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);
	} else if (nPlanes >768) {
		CrossMapNormalization_gradInput <1024> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);
	} else if (nPlanes >512) {
		CrossMapNormalization_gradInput <768> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);
	} else if (nPlanes >384) {
		CrossMapNormalization_gradInput <512> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);
	} else if (nPlanes >256) {
		CrossMapNormalization_gradInput <384> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);
	} else if (nPlanes >128) {
		CrossMapNormalization_gradInput <256> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);
	} else {
		CrossMapNormalization_gradInput <128> <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, iw, nPlanes, k, alpha, beta, n, str0, str1);
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
