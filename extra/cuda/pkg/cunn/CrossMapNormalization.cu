
/*
 * Description:
 *    this function crossmap-normalizes one single row
 *    along the innermost dimension
 *    Nd input, Nd output
 */
__global__ void CrossMapNormalization_output(float *input, float *output, float *z, long stride,
                           long nrows, long ncols, float k, float alpha, float beta, long n)
{
  // one thread = one row = one pixel (x size feature maps)
  long pixidx = threadIdx.x + blockDim.x * blockIdx.x;
  if (pixidx >= nrows) return;
  
  // input offset:
  long offset = pixidx;

  // move pointers
  input = input + offset;
  output = output + offset;  
  z = z + offset;

  float alphan = alpha / n;
  long i,j;
  long size = ncols;

   for(i = 0; i < size; i++)
       {
         float tmpz = 0;
         long startf = i - n/2;
         long endf = startf + n;
         startf = (startf < 0) ? 0 : startf;
         endf = (endf > size) ? size : endf;
         for(j=startf; j<endf; j++)
           {
             float x = input[j*stride];
             tmpz += x * x;
           }
         tmpz=pow(k+tmpz*alphan,-beta);
         z[i*stride] = tmpz;  /* z^{-beta}   */
         output[i*stride] = input[i*stride] * tmpz;
       }
  }





__global__ void CrossMapNormalization_gradInput(float *input, float* gradOutput, float* gradInput, float *z, long stride,
                              long nrows, long ncols, float k, float alpha, float beta, long n)
{
  long pixidx = threadIdx.x + blockDim.x * blockIdx.x;
  if (pixidx >= nrows) return;
  
  // input offset:
  long offset = pixidx;

  // move pointers
  input = input + offset;
  gradOutput = gradOutput + offset;  
  gradInput = gradInput + offset;  
  z = z + offset;

  float alphan = alpha / n;
  long i,j;
  long size = ncols;

      for(i = 0; i < size; i++)
       {
         float gradi = 0;
         float ai = input[i*stride];
         long endo = i + n/2 + 1;
         long starto = endo - n;
         starto = (starto < 0) ? 0 : starto;
         endo = (endo > size) ? size : endo;
         for (j=starto; j<endo; j++)
           {
             float aj = input[j*stride];
             float gj = gradOutput[j*stride];
             gradi += (i == j) ? gj * pow(z[j*stride], -beta) : 0;
             gradi -= gj * 2 * alphan * beta * ai * aj * pow(z[j*stride], -beta-1);
           }
         gradInput[i*stride]=gradi;
       }

}




static int cunn_CrossMapNormalization_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  int dimension = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *z = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "z", "torch.CudaTensor");
  float alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  float k = luaT_getfieldchecknumber(L, 1, "k");
  long n = luaT_getfieldcheckint(L, 1, "n");

  luaL_argcheck(L, dimension >= 0 && dimension < input->nDimension, 2, "dimension out of range");
  luaL_argcheck(L, dimension == 0, 2, "only supported dimension is first (MO)");

  input = THCudaTensor_newContiguous(input);
  

  THCudaTensor_resizeAs(output, input);
  THCudaTensor_resizeAs(z, input);

  float *input_data = THCudaTensor_data(input);
  float *output_data = THCudaTensor_data(output);
  float *z_data = THCudaTensor_data(z);

  long ncols = input->size[dimension];
  long nrows = THCudaTensor_nElement(output) / ncols;
  long stride = input->stride[dimension];

  // cuda blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);
  dim3 blocks(nblocks);
  dim3 threads(nthreads);

  // kernel:
  CrossMapNormalization_output <<<blocks, threads>>> (input_data, output_data, z_data, stride, nrows, ncols, k, alpha, beta, n);

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


static int cunn_CrossMapNormalization_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *z = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "z", "torch.CudaTensor");
  int dimension  = luaT_getfieldcheckint(L, 1, "dimension")-1;
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
  
  long ncols = input->size[dimension];
  long nrows = THCudaTensor_nElement(input) / ncols;
  long stride = input->stride[dimension];


  // cuda blocks & threads:
  long nthreads = 256;
  long nblocks = ceil((float)nrows / nthreads);
  dim3 blocks(nblocks);
  dim3 threads(nthreads);
  
  // kernel:
  CrossMapNormalization_gradInput <<<blocks, threads>>> (input_data, gradOutput_data, gradInput_data, z_data, stride, nrows, ncols, k, alpha, beta, n);

  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in CrossMapNormalization.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  return 1;
}

static const struct luaL_Reg cunn_CrossMapNormalization__ [] = {
  {"CrossMapNormalization_updateOutput", cunn_CrossMapNormalization_updateOutput},
  {"CrossMapNormalization_updateGradInput", cunn_CrossMapNormalization_updateGradInput},
  {NULL, NULL}
};

static void cunn_CrossMapNormalization_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunn_CrossMapNormalization__, "nn");
  lua_pop(L,1);
}
