#include <cublas_v2.h>
//#include "cusparse_v2.h"

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

#if 0
static int cunxn_testSgemm_cusparserun(lua_State *L)
{

  THCudaTensor *output = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THIntTensor *coorow = (THIntTensor *)luaT_checkudata(L, 3, "torch.IntTensor");
  THIntTensor *coocol = (THIntTensor *)luaT_checkudata(L, 4, "torch.IntTensor");
  THCudaTensor *cooval = (THCudaTensor *)luaT_checkudata(L, 5, "torch.CudaTensor");
  THCudaTensor *w      = (THCudaTensor *)luaT_checkudata(L, 6, "torch.CudaTensor");

  cusparseHandle_t handle =0;

  cusparseStatus_t err = cusparseCreate(&handle);
  if (err != CUSPARSE_STATUS_SUCCESS) {
    printf("error in creating handle");
  }

   
   // convert coorow to ints :
   
   float* csrValA   = THCudaTensor_data(cooval);

   float* wptr      = THCudaTensor_data(w);
   int m =  100000;
   int n = 2075;
   int k = 132809;
   
   int ldc = k;
   int ldb = 100000;

   float alpha=1;
   float beta=0;
   
   int sz=coorow->size[1];
   int nnz=sz;
   
   printf("nnz : %d\n", nnz);
   printf("ldb : %d\n", ldb);
   printf("step 1");
     
   int* coorowHptr  = THIntTensor_data(coorow);
   int* coocolHptr  = THIntTensor_data(coocol);
   int* csrColInd =0;
   cudaMalloc((void**)&csrColInd, sizeof(int) * sz);
   cudaMemcpy(csrColInd, coocolHptr, sizeof(int) * sz, cudaMemcpyHostToDevice);

   printf("step 2");
   
   int* cooRowInd =0;
   int* csrRowPtr =0;
   cudaMalloc((void**)&cooRowInd, sizeof(int) * sz);
   cudaMemcpy(cooRowInd, coorowHptr, sizeof(int) * sz, cudaMemcpyHostToDevice);
   err = cusparseXcoo2csr(handle, cooRowInd, sz, m, csrRowPtr, CUSPARSE_INDEX_BASE_ONE);
   cudaFree(cooRowInd);

   printf("step 3");

   cusparseMatDescr_t descrA;

   err = cusparseCreateMatDescr(&descrA);
   err = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE);
   if (err != CUSPARSE_STATUS_SUCCESS) {
    printf("error in descriptor");
   }
  
  cusparseOperation_t trans = CUSPARSE_OPERATION_TRANSPOSE;

   printf("step 4");
  
  //THCudaTensor_resize2d(output, ldb, m);
  //THCudaTensor_fill(output,0);
  float* optr = THCudaTensor_data(output);
  //int ldc=ldb;

   printf("step 5");
  
   err = cusparseScsrmm( handle, trans, m, n, k, nnz, &alpha, descrA, csrValA, csrRowPtr, csrColInd, wptr, ldb, &beta, optr, ldc);

   printf("step 6");


   if (err != CUSPARSE_STATUS_SUCCESS) {
    printf("error in csrmm");
   }

   if (err == CUSPARSE_STATUS_NOT_INITIALIZED) {
    printf("CUSPARSE_STATUS_NOT_INITIALIZED");
   }
               
   if (err == CUSPARSE_STATUS_ALLOC_FAILED) {
    printf("CUSPARSE_STATUS_ALLOC_FAILED");
   }
               
   if (err == CUSPARSE_STATUS_INVALID_VALUE) {
    printf("CUSPARSE_STATUS_INVALID_VALUE");
   }
               
   if (err == CUSPARSE_STATUS_ARCH_MISMATCH) {
    printf("CUSPARSE_STATUS_ARCH_MISMATCH");
   }
               
   if (err == CUSPARSE_STATUS_EXECUTION_FAILED) {
    printf("CUSPARSE_STATUS_EXECUTION_FAILED");
   }
               
   if (err == CUSPARSE_STATUS_INTERNAL_ERROR) {
    printf("CUSPARSE_STATUS_INTERNAL_ERROR");
   }
               
   if (err == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED) {
    printf("CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED");
   }
               
  err = cusparseDestroy(handle);
  if (err != CUSPARSE_STATUS_SUCCESS) {
    printf("error in destroying handle");
  }
return 0;
}
#endif

static const struct luaL_Reg cunxn_testSgemm__ [] = {
  {"testSgemm_run", cunxn_testSgemm_run},
//  {"testSgemm_cusparserun", cunxn_testSgemm_cusparserun},
  {NULL, NULL}
};

static void cunxn_testSgemm_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_testSgemm__, "nxn");
  lua_pop(L,1);
}
