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


__global__ void inputcopykernelsmall(float* inputptr, float* icopyptr, int stridey, int bs, int ih, 
      int iw, int ip, int padtop, int padleft, int toh, int tiw)
{
      /* blockIdx.z  = s     [ 0, stridey-1 ]
         blockIdx.y  = it1   [ 0, bs-1      ]
         blockIdx.x  = it3   [ 0, (iw/blockDim.y)-1+1      ]
         threadIdx.x = it4x  [ 0, ip-1      ]
         threadIdx.y = it4y  [ 0, 32/ip-1   ]
       */
         
      int fout = (MAX(0,padtop-blockIdx.z)+stridey-1)/stridey;
      int fin = fout * stridey - padtop + blockIdx.z;

      if (fin < ih) 
      {
         //inputptr += (blockIdx.y)*ih*iw*ip+fin*iw*ip+(blockIdx.x*blockDim.y)*ip;
         //icopyptr += blockIdx.z*bs*toh*tiw*ip+(blockIdx.y)*toh*tiw*ip+fout*tiw*ip+(padleft+blockIdx.x*blockDim.y)*ip;
         inputptr += (blockIdx.y)*ih*iw*ip+fin*iw*ip;
         icopyptr += blockIdx.z*bs*toh*tiw*ip+(blockIdx.y)*toh*tiw*ip+fout*tiw*ip+padleft*ip;
         
         int inputsize2   = ((ih-fin) + stridey - 1) / stridey;

         for (int it2=0; it2<inputsize2; it2++) { 
            //int iticopy3=blockIdx.z*bs*toh*tiw*ip+(blockIdx.y)*toh*tiw*ip+fout*tiw*ip+(padleft+blockIdx.x)*ip+it2*tiw*ip;
            //int itinput3=(blockIdx.y)*ih*iw*ip+fin*iw*ip+(blockIdx.x)*ip +it2*stridey*iw*ip;
            if((blockIdx.x*blockDim.y)*ip+threadIdx.x+blockDim.x*threadIdx.y<ip*iw) {
            icopyptr[(blockIdx.x*blockDim.y)*ip+threadIdx.x+blockDim.x*threadIdx.y]=inputptr[(blockIdx.x*blockDim.y)*ip+threadIdx.x+blockDim.x*threadIdx.y];
            }


            //for (int it4=threadIdx.x+blockDim.x*threadIdx.y; it4<ip*iw; it4+=blockDim.x*blockDim.y) 
            //{
            //   icopyptr[it4]=inputptr[it4];
            //}
            // => next row
            inputptr += stridey*iw*ip;
            icopyptr += tiw*ip;
			}
      }
}
      

__global__ void inputcopykernel(float* inputptr, float* icopyptr, int stridey, int bs, int ih, 
      int iw, int ip, int padtop, int padleft, int toh, int tiw)
{
      // blockIdx.z  = s     [ 0, stridey-1 ]
      // blockIdx.y  = it1   [ 0, bs-1      ]
      // blockIdx.x  = it3   [ 0, iw-1      ]
      // threadIdx.x = it4   [ 0, 31        ]
       
         
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
//  int oldpadright = padright;
//  int oldpadbottom = padbottom;
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

  //dim3 icopyblocks(iw, bs, stridey);
  // dim3 icopythreads(32);
  //dim3 icopythreads(MIN(32,ip), MAX(32/ip,1));
  
  if(ip<32) {
  dim3 icopyblocks(iw/(32/ip)+1, bs, stridey);
  dim3 icopythreads(MIN(32,ip), 32/ip);
  inputcopykernelsmall <<<icopyblocks, icopythreads>>> (inputptr, icopyptr, stridey, bs, ih, iw, ip, padtop, padleft, toh, tiw);
      /* blockIdx.z  = s     [ 0, stridey-1 ]
         blockIdx.y  = it1   [ 0, bs-1      ]
         blockIdx.x  = it3   [ 0, (iw/blockDim.y)-1+1      ]
         threadIdx.x = it4x  [ 0, ip-1      ]
         threadIdx.y = it4y  [ 0, 32/ip-1   ]
       */

  }
  else {
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











__global__ void kernelCopyReverse(float* weightptr, float* revkptr, int stridey, int stridex, int kouth, 
      int koutw, int kouto, int kouti, int sh, int so, int sw, int si, int kh, int kw, int ko, int ki)
{
   /*
         blockIdx.z  =    [ 0, ceil(ki/32)] -> usually this should be good
            inputplane = blockIdx.z * blockDim.x+threadIdx.x
         blockIdx.y  =    [ 0, stry-1    ]
         blockIdx.x  =    [ 0, strx-1    ]
         threadIdx.x =    [ 0, 31        ] -> weight input dim
         threadIdx.y =    [ 0, 31        ] -> weight output dim
            outputplane= iterator * blockDim.y + threadIdx.y
   */
   const int stry=blockIdx.y;
   const int strx=blockIdx.x;

   // put revkptr on proper stry,strx submatrix
   revkptr  +=    (blockIdx.y*stridex + blockIdx.x)*kouth*kouto*koutw*kouti;

   
   __shared__ float weightvalues[32][33];
   // for given x,y : weightvalues[inputplane][outputplane]
   
   
   int ith, itw, xcoord, ycoord, ito;
   for(ith=0; ith<kouth; ith++) {
      ycoord=kh-(ith*stridey+stry+1);
      if (ycoord<kh && ycoord>-1) {
         for(itw=0; itw<koutw; itw++) {
	   	   xcoord=kw-(itw*stridex+strx+1);
				if (xcoord<kw && xcoord>-1) {         

/*              int kh = weight->size[0];
              int op = weight->size[1];
              int kw = weight->size[2];
              assert(ip==weight->size[3]);            */
         
         for (ito=0; ito<(ko+blockDim.y-1)/blockDim.y; ito++) {

         /* iterate over tiles of size 32*32 */

         /* Step 1 : for a given (x,y)
            read weight(y, [32o], x, [32i]) and store the stuff in shmem */

                  const int curoplane=ito*blockDim.y+threadIdx.y;
                  const int curiplane=blockIdx.z * blockDim.x+threadIdx.x;
                  
                  if(curiplane<ki && curoplane<ko) {
                     weightvalues[threadIdx.x][threadIdx.y]=weightptr[ycoord*sh+xcoord*sw+(curoplane)*so+(curiplane)*si];
                  }
                  
                  __syncthreads();

         /* Step 2 : write revk(ith, [32i], itw, [32o]) in submatrix */
                  
                  const int reviplane=blockIdx.z * blockDim.y + threadIdx.y;
                  const int revoplane=ito*blockDim.x+threadIdx.x;
                  
                  if( reviplane < ki && revoplane < ko) {
                     revkptr[ith*kouto*koutw*kouti + itw*kouti + reviplane*koutw*kouti + revoplane] = weightvalues[threadIdx.y][threadIdx.x];
                  }

                  __syncthreads();

         }
         
         
   
         
         }
         }
      }
   }
   


   
}
      



__global__ void copyGradOut(float* goptr, float* gocpyptr, int goh, int gow, int pgoh, int pgow, int revkh, int revkw, int op)
{
   /* blockIdx.z  = [ 0, bs-1  ] (it1)
      blockIdx.y  = [ 0, goh-1 ] (it2)
      blockIdx.x  = [ 0, gow-1 ] (it3)
      threadIdx.x = [ 0, 31    ] (it4)
   */

   gocpyptr += ((blockIdx.z*pgoh+(revkh -1 + blockIdx.y))*pgow+(revkw-1+blockIdx.x))*op;
   goptr += ((blockIdx.z*goh+blockIdx.y)*gow+blockIdx.x)*op;

   int i;
   for(i=threadIdx.x; i<op; i+=blockDim.x)
   {
      gocpyptr[i]=goptr[i];
   }

}






__global__ void copyGradinResult(float* gradinptr, float* resptr, int throwawayx, int throwawayy, int stridey, int rs0, int rs1, int rs2, int gs0, int gs1, int gs2, int gs3, int ip, int gih, int padtop, int padleft, int ih, int iw)
{
   /*
      blockIdx.z  = [ 0, bs-1 ] (it1)
      blockIdx.y  = [ 0 ] 
      blockIdx.x  = [ 0, iw ] (it3)
      threadIdx.x = [ 0, 31   ] (it4)
   */

   int starty, sizey;
   
   resptr   += blockIdx.z*rs0 + blockIdx.x*rs2;
   gradinptr+= blockIdx.z*gs1 + (padleft + throwawayx + blockIdx.x)*gs3;
   
   float* tresptr ;
   float* tgradinptr;
   
   for(int stry=stridey; stry>0; stry--) {
   	int throwaway = stridey-stry < throwawayy;
	   if(throwaway) {
	   	starty = (stridey-stry+1) - throwawayy + stridey -1 ;
   		sizey  = gih-1;
   	}
	   else 	{ 
		   starty = (stridey-stry+1) - throwawayy -1 ;
		   sizey  = gih;
	   }
	   
	   for(int it2=0; it2<sizey; it2++) {
	      if((starty + it2*stridey - padtop)>-1 && (starty + it2*stridey - padtop)<ih)
	      {
            tresptr	   = resptr    + (starty + it2*stridey - padtop)*rs1;
            tgradinptr	= gradinptr + (stry-1)*gs0;
            if(throwaway)  { tgradinptr += (it2+1)*gs2 ; }
            else           { tgradinptr += it2*gs2 ; }
      
            for(int it4=threadIdx.x; it4<ip; it4+=blockDim.x)
            {
               tresptr[it4]=tgradinptr[it4];
            }
         }
      }
   }
   
}






static int cunxn_ConvProto_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *weight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
//  int dimension  = luaT_getfieldcheckint(L, 1, "dimension")-1;
  THCudaTensor *result  = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");
  THCudaTensor *revk;

  int stridex = luaT_getfieldcheckint(L, 1, "dW");
  int stridey = luaT_getfieldcheckint(L, 1, "dH");

  int padleft = luaT_getfieldcheckint(L, 1, "padleft");
  int padright = luaT_getfieldcheckint(L, 1, "padright");
  int padtop = luaT_getfieldcheckint(L, 1, "padtop");
  int padbottom = luaT_getfieldcheckint(L, 1, "padbottom");

  int overlap = luaT_getfieldcheckint(L, 1, "overlap");
  //assert(overlap==1);

  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  int bs = input->size[0];
  int ih = input->size[1];
  int iw = input->size[2];
  int ip = input->size[3];

  int kh = weight->size[0];
  int op = weight->size[1];
  int kw = weight->size[2];
  assert(ip==weight->size[3]);


   assert(gradOutput->nDimension == 4);
   assert(bs == gradOutput->size[0]);
   /* check that output h,w sizes match gradOutput sizes      */
   int goh = gradOutput->size[1];
   int gow = gradOutput->size[2];
   assert(goh == (ih + padtop + padbottom - kh) / stridey + 1) ;
   assert(gow == (iw + padleft + padright - kw) / stridex + 1) ;
   assert(op == gradOutput->size[3]);



   /*copyKernelReverse*/


   int ko = weight->size[1];
   int ki = weight->size[3];
   
   int sh = weight->stride[0];
   int so = weight->stride[1];
   int sw = weight->stride[2];
   int si = weight->stride[3];
   
   
   
   int kouth=(kh+stridey-1)/stridey;
   int kouto=ki;
   int koutw=(kw+stridex-1)/stridex;
   int kouti=ko;

   /* clean this after... */
   int revkh=kouth;
   int revkw=koutw;

   THLongStorage *revksize = THLongStorage_newWithSize(6);
   revksize->data[0]=stridey;
   revksize->data[1]=stridex;
   revksize->data[2]=kouth;
   revksize->data[3]=kouto;
   revksize->data[4]=koutw;
   revksize->data[5]=kouti;

   revk = THCudaTensor_newWithSize(revksize, NULL);
   THCudaTensor_fill(revk, 0);
   
   float* weightptr=THCudaTensor_data(weight);
   float* revkptr=THCudaTensor_data(revk);
   
   dim3 kcrblocks(stridex, stridey, (ki+31)/32);
   dim3 kcrthreads(32,32);
   
   
   kernelCopyReverse <<<kcrblocks, kcrthreads>>>(weightptr, revkptr, stridey, stridex, kouth, 
      koutw, kouto, kouti, sh, so, sw, si, kh, kw, ko, ki);
   /*
         blockIdx.z  =    [ 0, ceil(ki/32)] -> parallelizing over inputplanes dimension : 
            usually there will be lots of them except in data layer where there is no backprop
            inputplane = blockIdx.z * blockDim.x+threadIdx.x
         blockIdx.y  =    [ 0, stry-1    ]
         blockIdx.x  =    [ 0, strx-1    ]
         threadIdx.x =    [ 0, 31        ] -> weight input dim
         threadIdx.y =    [ 0, 31        ] -> weight output dim
            outputplane= iterator * blockDim.y + threadIdx.y
   */
   
   
   /* end of copyKernelReverse */
   
   
   
   /* create gradinput tensor :*/
   int giw = ( gow + revkw -1 ) * stridex;
   int gih = ( goh + revkh -1 ) ;

   THLongStorage *gradinsize = THLongStorage_newWithSize(5);
   gradinsize->data[0]=stridey;
   gradinsize->data[1]=bs;
   gradinsize->data[2]=gih;
   gradinsize->data[3]=giw;
   gradinsize->data[4]=ip;

   THCudaTensor * gradin = THCudaTensor_newWithSize(gradinsize, NULL);
   THCudaTensor_fill(gradin, 0);
   
   
   /* pad gradoutput tensor :*/
   int pgow = ( gow + revkw -1 );
   int pgoh = ( goh + revkh -1 );

   
   /* here we take bs+1 to have some zero-padding at the end of the matrix */
   /* it only costs some memory. GEMM does not use it. */

   THLongStorage *gradoutsize = THLongStorage_newWithSize(4);
   gradoutsize->data[0]=bs+1;
   gradoutsize->data[1]=pgoh;
   gradoutsize->data[2]=pgow;
   gradoutsize->data[3]=op;

   THCudaTensor * gradOutCopy = THCudaTensor_newWithSize(gradoutsize, NULL);
   THCudaTensor_fill(gradOutCopy, 0);

   float* goptr=THCudaTensor_data(gradOutput);
   float* gocpyptr=THCudaTensor_data(gradOutCopy);

   /* Convert this to a CUDA kernel 
   
   // Lua :
   gradOutCopy = newSameTensor(gradOutput, bs+1, pgoh, pgow, op) 
   tgocopy=narrowTensorAndZero(gradOutCopy, 1, 1, bs)
   tgocopy=narrowTensorAndZero(tgocopy, 2, revkh, goh)
   tgocopy=narrowTensorAndZero(tgocopy, 3, revkw, gow)
   tgocopy:copy(gradOutput)


   // C :
   real* goptr=THTensor_(data)(gradOutput);
   real* gocpyptr=THTensor_(data)(gradOutCopy);

   int itgocpy0=0;
   int itgo=0;

   int it1, it2, it3, it4;
   for (it1=0; it1<bs; it1++) {
		int itgocpy1	=	itgocpy0+(it1)*pgoh*pgow*op;
	    for (it2=0; it2<goh; it2++) { 
			int itgocpy2=itgocpy1+(revkh-1+it2)*pgow*op;
			for (it3=0; it3<gow; it3++ ) {
				int itgocpy3=itgocpy2+(revkw-1+it3)*op;
				for (it4=0; it4<op; it4++) {
					gocpyptr[itgocpy3]=goptr[itgo];
					itgocpy3++;
					itgo++;
				}
			}
		}
	} 

   
   */
   
   dim3 cgoblocks(gow, goh, bs);
   dim3 cgothreads(32);
   
   copyGradOut <<< cgoblocks, cgothreads >>>(goptr, gocpyptr, goh, gow, pgoh, pgow, revkh, revkw, op);

   /* blockIdx.z  = [ 0, bs-1  ] (it1)
      blockIdx.y  = [ 0, goh-1 ] (it2)
      blockIdx.x  = [ 0, gow-1 ] (it3)
      threadIdx.x = [ 0, 31    ] (it4)
   */
   
   /* end of copyGradOut */
   
   float onef=1;
   
  cublasHandle_t handle;
  cublasStatus_t err = cublasCreate(&handle);
  if (err != CUBLAS_STATUS_SUCCESS) { printf("error in creating handle"); }
   
   /* GEMM calls : */
	int nxs=1;
	if(!overlap) {
	   nxs=revkw; 
	   //printf("no overlap");
	}
	for (int hcall=0; hcall<nxs; hcall++) {
	   for (int stry=0; stry<stridey; stry++) {
		   for (int strx=0; strx<stridex; strx++) {
			   for (int vcall=0; vcall<revkh; vcall++) {
				   float* gradoutptr  = THCudaTensor_data(gradOutCopy);
				   gradoutptr		   += (revkh-vcall-1)*gradOutCopy->stride[1] + hcall*gradOutCopy->stride[2];
               int ldgradout      = op*nxs;
                     
				   float* krevptr	    = THCudaTensor_data(revk);
				   krevptr 		      += (stry)*revk->stride[0] + (strx)*revk->stride[1] + (revkh-vcall-1)*revk->stride[2];
               int szkrev         = op*revkw;
               int ldkrev     	 = op*revkw;
                  
				   float* gradinptr	 = THCudaTensor_data(gradin);
				   gradinptr		+= (stry)*gradin->stride[0] + (stridex-(strx)-1+hcall*stridex)*gradin->stride[3];
               int ldgradin   	 = ip * stridex * nxs;
                  
               int nspots         = giw/stridex*gih*bs;
               int ngem           = (nspots-hcall+nxs-1)/nxs;
                  
               err = cublasSgemm(handle,
                           CUBLAS_OP_T, CUBLAS_OP_N,
                           ip, ngem, szkrev,
                           &onef,
                           krevptr, ldkrev,
                           gradoutptr, ldgradout,
                           &onef,
                           gradinptr, ldgradin );

               if (err != CUBLAS_STATUS_SUCCESS) { printf("error in sgemm"); }
               //else {printf("called sgemm..."); }
			   }
		   }
	   }
   }

  err = cublasDestroy(handle);
  if (err != CUBLAS_STATUS_SUCCESS) { printf("error in destroying handle"); }
   
   
   
   
   
   
   
   
   
   
   
   
     /* correct padright and padbottom */
  //int oldpadright = padright;
  //int oldpadbottom = padbottom;
  padright = gow * stridex + kw - stridex - iw - padleft;
  padbottom = goh * stridey + kh - stridey - ih - padtop;
  /* assert(not exact or padright ~= oldpadright, "horizontal size mismatch"); */
  /* assert(not exact or padbottom ~= oldpadbottom, "horizontal size mismatch"); */
  if (padright < 0)  { padright = 0;}
  if (padbottom < 0) { padbottom = 0;}

  /* input size with padding */
  //int piw = padleft + iw + padright; 
  //int pih = padtop + ih + padbottom;



    
   int throwawayx=stridex - kw%stridex;
   int throwawayy=stridey - kh%stridey;
   if (stridex==1 || stridex==throwawayx) { throwawayx=0 ; } 
   if (stridey==1 || stridey==throwawayy) { throwawayy=0 ; }

   /* clean this after */ 
   int resw=iw;
   int resh=ih;

   THCudaTensor_resize4d(result, bs, resh, resw, ip);
   THCudaTensor_fill(result, 0);


   float* gradinptr = THCudaTensor_data(gradin);
   float* resptr = THCudaTensor_data(result);

   /* Convert this to a CUDA kernel 
   int itres0 = 0;
   int itgi0  = 0;
   int starty, sizey;

      for(stry=stridey; stry>0; stry--) {
      	int throwaway = stridey-stry < throwawayy;
	      if(throwaway) {
		      starty = (stridey-stry+1) - throwawayy + stridey -1 ;
		      sizey  = gih-1;
       	}
	      else 	{ 
		      starty = (stridey-stry+1) - throwawayy -1 ;
		      sizey  = gih;
	      }

	      itgi0 = (stry-1)*gradin->stride[0];
	
         for (it1=0; it1<bs; it1++) {
		      int itres1 = itres0 + it1*result->stride[0];
		      int itgi1  = itgi0  + it1*gradin->stride[1];
		      for (it2=0; it2<sizey; it2++) { 
			      int itres2 = itres1 + (starty + it2*stridey)*result->stride[1];
			      int itgi2  = itgi1 + it2*gradin->stride[2];
			      if(throwaway) {itgi2 += gradin->stride[2];}
			      for (it3=0; it3<giw-throwawayx; it3++ ) {
				      int itres3 = itres2 + it3*result->stride[2];
				      int itgi3  = itgi2 + (throwawayx+it3)*gradin->stride[3];
				      for (it4=0; it4<ip; it4++) {
					      resptr[itres3]= gradinptr[itgi3];
					      itres3++;
					      itgi3++;
				      }
			      }
		      }
	      } 


      }

   */



   dim3 cgirblocks(iw, 1, bs);
   dim3 cgirthreads(32);

   int rs0 = result->stride[0];
   int rs1 = result->stride[1];
   int rs2 = result->stride[2];
   int gs0 = gradin->stride[0]; 
   int gs1 = gradin->stride[1];
   int gs2 = gradin->stride[2];
   int gs3 = gradin->stride[3];
 

   copyGradinResult <<<cgirblocks,cgirthreads>>> (gradinptr, resptr, throwawayx, throwawayy, stridey, rs0, rs1, rs2, gs0, gs1, gs2, gs3, ip, gih, padtop, padleft, ih, iw);
   /*
      blockIdx.z  = [ 0, bs-1 ] (it1)
      blockIdx.y  = [ 0 ] 
      blockIdx.x  = [ 0, iw ] (it3)
      threadIdx.x = [ 0, 31   ] (it4)
   */



   /*real* gradinputptr = THTensor_(data)(gradInput);

   itgi0=0;
   itres0=0;
   for (it1=0; it1<bs; it1++) {
		int itres1 = itres0 + it1*result->stride[0];
		int itgi1  = itgi0  + it1*gradInput->stride[0];
		for (it2=0; it2<ih; it2++) { 
			int itres2 = itres1 + (padtop + it2)*result->stride[1];
			int itgi2  = itgi1 + it2*gradInput->stride[1];
			for (it3=0; it3<iw; it3++ ) {
				int itres3 = itres2 + (padleft+it3)*result->stride[2];
				int itgi3  = itgi2 + it3*gradInput->stride[2];
				for (it4=0; it4<ip; it4++) {
					gradinputptr[itgi3]= resptr[itres3];
					itres3++;
					itgi3++;
				}
			}
		}
	} */

   
   
   THCudaTensor_free(gradin);
   THCudaTensor_free(revk);
   THCudaTensor_free(gradOutCopy);
   
   
   
   
   
   //THCudaTensor_resizeAs(gradInput, result);
   //THCudaTensor_copy(gradInput, result);
   
   
      

  // check for errors
  cudaError_t err2 = cudaGetLastError();
  if (err2 != cudaSuccess) {
    printf("error in ConvProto.updateOutput: %s\n", cudaGetErrorString(err2));
    THError("aborting");
  }

  return 1;
}






__global__ void copyGradOutInBuffer(float* goptr, float* gocpyptr, int oh, int ow, int toh, int tow, int op)
{
   /* blockIdx.z  = [ 0, bs-1  ] (it1)
      blockIdx.y  = [ 0, oh-1  ] (it2)
      blockIdx.x  = [ 0, ow-1  ] (it3)
      threadIdx.x = [ 0, 31    ] (it4)
   */

   gocpyptr += blockIdx.z*toh*tow*op + blockIdx.y*tow*op + blockIdx.x*op;
   goptr += ((blockIdx.z*oh+blockIdx.y)*ow+blockIdx.x)*op;

   int i;
   for(i=threadIdx.x; i<op; i+=blockDim.x)
   {
      gocpyptr[i]=goptr[i];
   }

}




__global__ void computeGradBias(float* goptr, float* gradbiasptr, int bs, int oh, int ow, int op, float scale)
{
   /* blockIdx.x  = [ 0, ceil(op/32) ]
      blockIdx.y  = [ 0, bs-1        ]
      threadIdx.x = [ 0, 31          ]   
   */

   goptr += blockIdx.y*oh*ow*op;
   const int idx = blockIdx.x * blockDim.x + threadIdx.x;
   
   float b=0;
   
   if (idx<op) {
      for(int i=0; i<oh*ow; i++) {
         b += goptr[i*op + idx];
      }
   atomicAdd(&gradbiasptr[idx], b*scale);
   }
   
}






static int cunxn_ConvProto_accGradParameters(lua_State *L)
{



  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");

  float scale = luaL_optnumber(L, 4, 1);

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

  int kh = gradWeight->size[0];
  int op = gradWeight->size[1];
  int kw = gradWeight->size[2];
  assert(ip==gradWeight->size[3]);
  
  /* compute output size */
  int ow = ( iw + padleft + padright - kw ) / stridex + 1;
  int oh = ( ih + padtop + padbottom - kh ) / stridey + 1;

  /* correct padright and padbottom */
//  int oldpadright = padright;
//  int oldpadbottom = padbottom;
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

  if(ip<32) {
  dim3 icopyblocks(iw/(32/ip)+1, bs, stridey);
  dim3 icopythreads(MIN(32,ip), 32/ip);
  inputcopykernelsmall <<<icopyblocks, icopythreads>>> (inputptr, icopyptr, stridey, bs, ih, iw, ip, padtop, padleft, toh, tiw);
  }
  else {
  dim3 icopyblocks(iw, bs, stridey);
  dim3 icopythreads(32);
  inputcopykernel <<<icopyblocks, icopythreads>>> (inputptr, icopyptr, stridey, bs, ih, iw, ip, padtop, padleft, toh, tiw);
  }


  THCudaTensor* kcopy = gradWeight;
  THCudaTensor* ocopy = THCudaTensor_newWithSize4d(bs, toh, tow, op);
  THCudaTensor_fill(ocopy, 0);
  
  float* gradoutptr=THCudaTensor_data(gradOutput);
  float* ocpyptr=THCudaTensor_data(ocopy);
  
  dim3 goibblocks(ow, oh, bs);
  dim3 goibthreads(32);
  
   copyGradOutInBuffer <<<goibblocks,goibthreads>>>(gradoutptr, ocpyptr, oh, ow, toh, tow, op);

   /* blockIdx.z  = [ 0, bs-1  ] (it1)
      blockIdx.y  = [ 0, oh-1  ] (it2)
      blockIdx.x  = [ 0, ow-1  ] (it3)
      threadIdx.x = [ 0, 31    ] (it4)
   */


  float* gradbiasptr=THCudaTensor_data(gradBias);
  
  dim3 gbblocks((op+31)/32, bs);
  dim3 gbthreads(32);
  computeGradBias <<< gbblocks, gbthreads >>> (gradoutptr, gradbiasptr, bs, oh, ow, op, scale);
  
   /* blockIdx.x  = [ 0, ceil(op/32) ]
      threadIdx.x = [ 0, 31          ]   
   */


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
                           CUBLAS_OP_N, CUBLAS_OP_T,
                           kw*ip,op, ngem, 
                           &scale,
                           iptr, nxs*stridex*ip, 
                           optr, nxs*op, 
                           &onef,
                           kptr, kw*ip );     
              
              
              
         if (err != CUBLAS_STATUS_SUCCESS) { printf("error in sgemm"); }
         //else {printf("called sgemm..."); }
      }
   }


  err = cublasDestroy(handle);
  if (err != CUBLAS_STATUS_SUCCESS) { printf("error in destroying handle"); }





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














static const struct luaL_Reg cunxn_ConvProto__ [] = {
  {"ConvProto_updateOutput", cunxn_ConvProto_updateOutput},
  {"ConvProto_updateGradInput", cunxn_ConvProto_updateGradInput},
  {"ConvProto_accGradParameters", cunxn_ConvProto_accGradParameters},
  {NULL, NULL}
};

static void cunxn_ConvProto_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_ConvProto__, "nxn");
  lua_pop(L,1);
}
