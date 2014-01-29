#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) < (Y) ? (Y) : (X))






/********************************************/


__global__ void inputcopykernel(float* inputptr, float* icopyptr, int stridey, int bs, int ih, 
      int iw, int ip, int padtop, int padleft, int toh, int tiw)
{
      /* blockIdx.z  = s     [ 0, stridey-1 ]
         blockIdx.y  = it1   [ 0, bs-1      ]
         blockIdx.x  = it3   [ 0, iw-1      ]
         threadIdx.x = it4   [ 0, 31        ]
       */
         
      int fout = (MAX(0,padtop-blockIdx.z)+stridey-1)/stridey;
      int fin = fout * stridey - padtop + blockIdx.z;

      if (fin < ih) 
      {
         inputptr += (blockIdx.y)*ih*iw*ip+fin*iw*ip+(blockIdx.x)*ip;
         icopyptr += blockIdx.z*bs*toh*tiw*ip+(blockIdx.y)*toh*tiw*ip+fout*tiw*ip+(padleft+blockIdx.x)*ip;
         
         int inputsize2   = ((ih-fin) + stridey - 1) / stridey;

         for (int it2=0; it2<inputsize2; it2++) { 
            //int iticopy3=blockIdx.z*bs*toh*tiw*ip+(blockIdx.y)*toh*tiw*ip+fout*tiw*ip+(padleft+blockIdx.x)*ip+it2*tiw*ip;
            //int itinput3=(blockIdx.y)*ih*iw*ip+fin*iw*ip+(blockIdx.x)*ip +it2*stridey*iw*ip;

            for (int it4=threadIdx.x; it4<ip; it4+=blockDim.x) 
            {
               icopyptr[it4]=inputptr[it4];
            }
            
            inputptr += stridey*iw*ip;
            icopyptr += tiw*ip;
			}
      }
}
      


/* Convert this to a CUDA kernel...

  // Lua version of what happens, so we know what we're doing...
      for s=0,stridey-1 do
         local ticopy = icopy:select(1,s+1)
         local fout = math.floor((math.max(0,padtop-s)+stridey-1)/stridey)
         local fin = fout * stridey - padtop + s
         assert(fout >= 0 and fin >= 0)
         if fin < ih then
            local tinput = input:narrow(2,fin+1,ih-fin)
            local tinputSizes = tinput:size()
            local tinputStrides = tinput:stride()
            tinputStrides[2] = tinputStrides[2] * stridey
            tinputSizes[2] = math.floor((tinputSizes[2] + stridey - 1) / stridey)
            tinput = tinput.new(tinput:storage(), tinput:storageOffset(), tinputSizes, tinputStrides)
            ticopy = narrowTensorAndZero(ticopy, 2, fout+1, tinput:size(2))
            ticopy = narrowTensorAndZero(ticopy, 3, padleft+1, tinput:size(3))
            ticopy:copy(tinput)
         else
            ticopy:zero()
         end
      end

***********************************

int s;
   for (s=0; s<stridey; s++) {
      int fout = (MAX(0,padtop-s)+stridey-1)/stridey;
      int fin = fout * stridey - padtop + s;
      assert(fout >= 0 && fin >= 0);
      real* icopyptr=THTensor_(data)(icopy);
      real* inputptr=THTensor_(data)(input);

      if (fin < ih) {
         int inputsize2   = ((ih-fin) + stridey - 1) / stridey;
         int iticopy0=s*bs*toh*tiw*ip;
         int itinput0=0;
         int it1;
         for (it1=0; it1<bs; it1++) {
            int iticopy1=iticopy0+(it1)*toh*tiw*ip;
            int itinput1=itinput0+(it1)*ih*iw*ip;
            int it2;

            for (it2=0; it2<inputsize2; it2++) { 
               int iticopy2=iticopy1+(fout+it2)*tiw*ip;
               int itinput2=itinput1+((fin+1)+(it2)*stridey-1)*iw*ip;
               int it3;
               for (it3=0; it3<iw; it3++ ) {
                  int iticopy3=iticopy2+(padleft+it3)*ip;
                  int itinput3=itinput2+(it3)*ip;
                  int it4;
                  for (it4=0; it4<ip; it4++) {
                     icopyptr[iticopy3]=inputptr[itinput3];
                     iticopy3++;
                     itinput3++;
					}
				}
			}
		 } 
	  }
      else {
         int foo=bs*toh*tiw*ip*(s+1);
		 int it;
         for (it=bs*toh*tiw*ip*s; it<foo; it++){
            icopyptr[it]=0;
         }
      }
   }
*/







/********************************************/


__global__ void outputcopykernel(float* outputptr, float* ocopyptr, float* biasptr, int bs, int oh, 
      int ow, int op, int toh, int tow, float alpha, float beta) //alpha=1, beta=0
      {
      /* blockIdx.z  = it1   [ 0, bs-1      ]
         blockIdx.y  = it2   [ 0, oh-1      ]
         blockIdx.x  = it3   [ 0, ow-1      ]
         threadIdx.x = it4   [ 0, 31        ]
       */      
         outputptr += (blockIdx.z)*oh*ow*op+(blockIdx.y)*ow*op+(blockIdx.x)*op;
         ocopyptr  += (blockIdx.z)*toh*tow*op+(blockIdx.y)*tow*op+(blockIdx.x)*op;
         for (int it4=threadIdx.x; it4<op; it4+=blockDim.x) {
            outputptr[it4]=ocopyptr[it4] + biasptr[it4];
		   }
      }


/********************************************/


/* Convert this to a CUDA kernel...

  // Lua version of what happens, so we know what we're doing...
   -- accumulate output chunk into result tensor
   result:resize(bs,oh,ow,op)
   local tocopy = ocopy:narrow(2,1,oh):narrow(3,1,ow)
   if beta == 0 and alpha == 1 then
      result:copy(tocopy)
   elseif beta == 1 then
      result.add(tocopy, alpha)
   else
      result.mul(beta)
      result.add(tocopy, value)
   end
   
***********************************   

  THTensor_(resize4d)(output, bs, oh, ow, op);

  real* ocpyptr = THTensor_(data)(ocopy);
  real* optr = THTensor_(data)(output);
  real* bptr = THTensor_(data)(bias);

  // here we take alpha = 1 and beta = 0 
	int itout0=0;
	int itocpy0=0;
	int it1;
         for (it1=0; it1<bs; it1++) {
            int itout1=itout0+(it1)*oh*ow*op;
            int itocpy1=itocpy0+(it1)*toh*tow*op;
            int it2;
            for (it2=0; it2<oh; it2++) { 
               int itout2=itout1+(it2)*ow*op;
               int itocpy2=itocpy1+(it2)*tow*op;
               int it3;
               for (it3=0; it3<ow; it3++ ) {
                  int itout3=itout2+(it3)*op;
                  int itocpy3=itocpy2+(it3)*op;
                  int it4;
                  for (it4=0; it4<op; it4++) {
                     optr[itout3]=ocpyptr[itocpy3] + bptr[it4];
                     itout3++;
                     itocpy3++;
					}
				}
			}
		 } 

*/


/********************************************/

static int cunxn_ConvProto_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");

  int stridex = luaT_getfieldcheckint(L, 1, "dW");
  int stridey = luaT_getfieldcheckint(L, 1, "dH");

  int padleft = luaT_getfieldcheckint(L, 1, "padleft");
  int padright = luaT_getfieldcheckint(L, 1, "padright");
  int padtop = luaT_getfieldcheckint(L, 1, "padtop");
  int padbottom = luaT_getfieldcheckint(L, 1, "padbottom");

  int overlap = luaT_getfieldcheckint(L, 1, "overlap");

  float alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  float beta = luaT_getfieldchecknumber(L, 1, "beta");
  
  float onef=1;


  int bs = input->size[0];
  int ih = input->size[1];
  int iw = input->size[2];
  int ip = input->size[3];

  int kh = weight->size[0];
  int op = weight->size[1];
  int kw = weight->size[2];
  assert(ip==weight->size[3]);
  
  /* compute output size */
  int ow = ( iw + padleft + padright - kw ) / stridex + 1;
  int oh = ( ih + padtop + padbottom - kh ) / stridey + 1;

  /* correct padright and padbottom */
  int oldpadright = padright;
  int oldpadbottom = padbottom;
  padright = ow * stridex + kw - stridex - iw - padleft;
  padbottom = oh * stridey + kh - stridey - ih - padtop;
  /* assert(not exact or padright ~= oldpadright, "horizontal size mismatch"); */
  /* assert(not exact or padbottom ~= oldpadbottom, "horizontal size mismatch"); */
  if (padright < 0)  { padright = 0;}
  if (padbottom < 0) { padbottom = 0;}

  /* input size with padding */
  int piw = padleft + iw + padright; 
  int pih = padtop + ih + padbottom;

  /* number of horizontal strides between nonoverlapping runs */
  int nxs = 1;
  if (!overlap) { nxs = (kw + stridex - 1) / stridex ;}

  /* total size of output buffer */
  int tow = (piw + stridex - 1) / stridex;
  int toh = (pih + stridey - 1) / stridey;

  /* total size of input and output buffers */
  int tiw = tow * stridex;
  int tih = toh * stridey;  
  assert(tiw >= piw && piw >= iw);
  assert(tih >= pih && pih >= ih);

  /*icopy =  newSameTensor(input, stridey, bs, toh, tiw, ip) */
  THLongStorage *icopysize = THLongStorage_newWithSize(5);
  icopysize->data[0]=stridey;
  icopysize->data[1]=bs;
  icopysize->data[2]=toh;
  icopysize->data[3]=tiw;
  icopysize->data[4]=ip;
  THCudaTensor* icopy = THCudaTensor_newWithSize(icopysize, NULL);
  THCudaTensor_fill(icopy, 0);


  float* icopyptr=THCudaTensor_data(icopy);
  float* inputptr=THCudaTensor_data(input);

  dim3 icopyblocks(iw, bs, stridey);
  dim3 icopythreads(32);
  
  inputcopykernel <<<icopyblocks, icopythreads>>> (inputptr, icopyptr, stridey, bs, ih, iw, ip, padtop, padleft, toh, tiw);

/*__global__ void inputcopykernel(float* inputptr, float* icopyptr, int stridey, int bs, int ih, 
      int iw, int ip, int padtop, int padleft, int toh, int tiw)
         blockIdx.z  = s     [ 0, stridey-1 ]
         blockIdx.y  = it1   [ 0, bs-1      ]
         blockIdx.x  = it3   [ 0, iw-1      ]
         threadIdx.x = it4   [ 0, 31        ]
*/
       
       
/* Convert this to a CUDA kernel...

  // Lua version of what happens, so we know what we're doing...
      for s=0,stridey-1 do
         local ticopy = icopy:select(1,s+1)
         local fout = math.floor((math.max(0,padtop-s)+stridey-1)/stridey)
         local fin = fout * stridey - padtop + s
         assert(fout >= 0 and fin >= 0)
         if fin < ih then
            local tinput = input:narrow(2,fin+1,ih-fin)
            local tinputSizes = tinput:size()
            local tinputStrides = tinput:stride()
            tinputStrides[2] = tinputStrides[2] * stridey
            tinputSizes[2] = math.floor((tinputSizes[2] + stridey - 1) / stridey)
            tinput = tinput.new(tinput:storage(), tinput:storageOffset(), tinputSizes, tinputStrides)
            ticopy = narrowTensorAndZero(ticopy, 2, fout+1, tinput:size(2))
            ticopy = narrowTensorAndZero(ticopy, 3, padleft+1, tinput:size(3))
            ticopy:copy(tinput)
         else
            ticopy:zero()
         end
      end
      
      


int s;
   for (s=0; s<stridey; s++) {
      int fout = (MAX(0,padtop-s)+stridey-1)/stridey;
      int fin = fout * stridey - padtop + s;
      assert(fout >= 0 && fin >= 0);
      real* icopyptr=THTensor_(data)(icopy);
      real* inputptr=THTensor_(data)(input);

      if (fin < ih) {
         int inputsize2   = ((ih-fin) + stridey - 1) / stridey;
         int iticopy0=s*bs*toh*tiw*ip;
         int itinput0=0;
         int it1;
         for (it1=0; it1<bs; it1++) {
            int iticopy1=iticopy0+(it1)*toh*tiw*ip;
            int itinput1=itinput0+(it1)*ih*iw*ip;
            int it2;

            for (it2=0; it2<inputsize2; it2++) { 
               int iticopy2=iticopy1+(fout+it2)*tiw*ip;
               int itinput2=itinput1+((fin+1)+(it2)*stridey-1)*iw*ip;
               int it3;
               for (it3=0; it3<iw; it3++ ) {
                  int iticopy3=iticopy2+(padleft+it3)*ip;
                  int itinput3=itinput2+(it3)*ip;
                  int it4;
                  for (it4=0; it4<ip; it4++) {
                     icopyptr[iticopy3]=inputptr[itinput3];
                     iticopy3++;
                     itinput3++;
					}
				}
			}
		 } 
	  }
      else {
         int foo=bs*toh*tiw*ip*(s+1);
		 int it;
         for (it=bs*toh*tiw*ip*s; it<foo; it++){
            icopyptr[it]=0;
         }
      }
   }
*/



  THCudaTensor* kcopy = weight;
  THCudaTensor* ocopy = THCudaTensor_newWithSize4d(bs, toh, tow, op);
  THCudaTensor_fill(ocopy, 0);


  cublasHandle_t handle;
  cublasStatus_t err = cublasCreate(&handle);
  if (err != CUBLAS_STATUS_SUCCESS) { printf("error in creating handle"); }


   /* call GEMM */
	int hcall;
   for (hcall=0; hcall<nxs; hcall++) {
	   int vcall;
      for (vcall=0; vcall<kh; vcall++) {
         int sq = vcall / stridey;
         int sr = vcall - sq * stridey;
         /* local icopy =  newSameTensor(input, stridey, bs, toh, tiw, ip) */
         /* float* iptr = torch.data(icopy[{sr+1,{},sq+1,hcall*stridex+1,{}}]) */
		   float* iptr = THCudaTensor_data(icopy);
		   iptr       += (sr)*icopy->stride[0] + (sq)*icopy->stride[2] +  (hcall*stridex)*icopy->stride[3];

         /* local kptr  = torch.data(kcopy:select(1,vcall+1)) */
		   float* kptr = THCudaTensor_data(kcopy);
		   kptr	 	+= vcall * kcopy->stride[0];

         /* local optr = torch.data(ocopy:select(3,hcall+1)) */
		   float* optr = THCudaTensor_data(ocopy);
         optr		+= hcall * ocopy->stride[2];


         int nrun = (bs-1)*toh*tow + oh*tow;
         int ngem = (nrun - hcall) / nxs;

         //printf("calling sgemm...");

         /*THBlas_(gemm)('T','N', op, ngem, kw*ip, 
              1, kptr, kw*ip, iptr, nxs*stridex*ip,
              1, optr, nxs*op ); */
              
         err = cublasSgemm(handle,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           op, ngem, kw*ip,
                           &onef,
                           kptr, kw*ip,
                           iptr, nxs*stridex*ip,
                           &onef,
                           optr, nxs*op );     
              
              
              
         if (err != CUBLAS_STATUS_SUCCESS) { printf("error in sgemm"); }
         //else {printf("called sgemm..."); }
      }
   }

  err = cublasDestroy(handle);
  if (err != CUBLAS_STATUS_SUCCESS) { printf("error in destroying handle"); }



  THCudaTensor_resize4d(output, bs, oh, ow, op);

  float* ocopyptr=THCudaTensor_data(ocopy);
  float* outputptr=THCudaTensor_data(output);
  float* biasptr=THCudaTensor_data(bias);

  dim3 ocopyblocks(ow, oh, bs);
  dim3 ocopythreads(32);
  
  outputcopykernel <<<ocopyblocks, ocopythreads>>> (outputptr, ocopyptr, biasptr, bs, oh, ow, op, toh, tow, alpha, beta);
  // alpha and beta are actually not used

/*__global__ void outputcopykernel(float* outputptr, float* ocopyptr, float* biasptr, int bs, int oh, 
      int ow, int op, int toh, int tow, float alpha, float beta) //alpha=1, beta=0
         blockIdx.z  = it1   [ 0, bs-1      ]
         blockIdx.y  = it2   [ 0, oh-1      ]
         blockIdx.x  = it3   [ 0, ow-1      ]
         threadIdx.x = it4   [ 0, 31        ]
*/      


/* Convert this to a CUDA kernel...

  // Lua version of what happens, so we know what we're doing...
   -- accumulate output chunk into result tensor
   result:resize(bs,oh,ow,op)
   local tocopy = ocopy:narrow(2,1,oh):narrow(3,1,ow)
   if beta == 0 and alpha == 1 then
      result:copy(tocopy)
   elseif beta == 1 then
      result.add(tocopy, alpha)
   else
      result.mul(beta)
      result.add(tocopy, value)
   end
   
   

  THTensor_(resize4d)(output, bs, oh, ow, op);

  real* ocpyptr = THTensor_(data)(ocopy);
  real* optr = THTensor_(data)(output);
  real* bptr = THTensor_(data)(bias);

  // here we take alpha = 1 and beta = 0 
	int itout0=0;
	int itocpy0=0;
	int it1;
         for (it1=0; it1<bs; it1++) {
            int itout1=itout0+(it1)*oh*ow*op;
            int itocpy1=itocpy0+(it1)*toh*tow*op;
            int it2;
            for (it2=0; it2<oh; it2++) { 
               int itout2=itout1+(it2)*ow*op;
               int itocpy2=itocpy1+(it2)*tow*op;
               int it3;
               for (it3=0; it3<ow; it3++ ) {
                  int itout3=itout2+(it3)*op;
                  int itocpy3=itocpy2+(it3)*op;
                  int it4;
                  for (it4=0; it4<op; it4++) {
                     optr[itout3]=ocpyptr[itocpy3] + bptr[it4];
                     itout3++;
                     itocpy3++;
					}
				}
			}
		 } 

*/




//THCudaTensor_resizeAs(output, ocopy);
//THCudaTensor_copy(output, ocopy);



  // check for errors
  cudaError_t lasterror = cudaGetLastError();
  if (lasterror != cudaSuccess) {
    printf("error in ConvProto.updateOutput: %s\n", cudaGetErrorString(lasterror));
    THError("aborting");
  }
 
  // final cut:
  //THCudaTensor_free(input); 
  THCudaTensor_free(icopy);
  THCudaTensor_free(ocopy);
  //THCudaTensor_select(output, NULL, dimension, 0);

  return 1;
}











static int cunxn_ConvProto_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *z = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "z", "torch.CudaTensor");
//  int dimension  = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *gradInput  = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in ConvProto.updateOutput: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  return 1;
}

static const struct luaL_Reg cunxn_ConvProto__ [] = {
  {"ConvProto_updateOutput", cunxn_ConvProto_updateOutput},
  {"ConvProto_updateGradInput", cunxn_ConvProto_updateGradInput},
  {NULL, NULL}
};

static void cunxn_ConvProto_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_ConvProto__, "nxn");
  lua_pop(L,1);
}
