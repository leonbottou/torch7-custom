#include <cublas_v2.h>

static int cunxn_testSgemm_run(lua_State *L)
{
  //THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *A = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *B = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *C = (THCudaTensor *)luaT_checkudata(L, 4, "torch.CudaTensor");

  long m = luaT_getfieldcheckint(L, 1, "m");
  long n = luaT_getfieldcheckint(L, 1, "n");
  long k = luaT_getfieldcheckint(L, 1, "k");

  long lda = luaT_getfieldcheckint(L, 1, "lda");
  long ldb = luaT_getfieldcheckint(L, 1, "ldb");
  long ldc = luaT_getfieldcheckint(L, 1, "ldc");

  long tA = luaT_getfieldcheckint(L, 1, "tA");
  long tB = luaT_getfieldcheckint(L, 1, "tB");

  cublasOperation_t transa=CUBLAS_OP_N;
  cublasOperation_t transb=CUBLAS_OP_N;


	if(tA==1) { transa=CUBLAS_OP_T; }
	if(tB==1) { transb=CUBLAS_OP_T; }

  float alpha = 1.0f;
  float beta = 1.0f;


  float* ptrA  = THCudaTensor_data(A);
  float* ptrB   = THCudaTensor_data(B);
  float* ptrC    = THCudaTensor_data(C);

	cublasHandle_t handle;


  cublasStatus_t err = cublasCreate(&handle);
  if (err != CUBLAS_STATUS_SUCCESS) {
    printf("error in creating handle");
  }

printf("calling sgemm...");
  err = cublasSgemm(handle,
                           transa, transb,
                           m, n, k,
                           &alpha,
                           ptrA, lda,
                           ptrB, ldb,
                           &beta,
                           ptrC, ldc);
  if (err != CUBLAS_STATUS_SUCCESS) {
    printf("error in sgemm");
  }
printf("called sgemm...");

err = cublasDestroy(handle);
  if (err != CUBLAS_STATUS_SUCCESS) {
    printf("error in destroying handle");
  }



  return 1;
}



static const struct luaL_Reg cunxn_testSgemm__ [] = {
  {"testSgemm_run", cunxn_testSgemm_run},
  {NULL, NULL}
};

static void cunxn_testSgemm_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_testSgemm__, "nxn");
  lua_pop(L,1);
}
