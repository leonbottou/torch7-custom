#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/ConvProto.c"
#else

#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) < (Y) ? (Y) : (X))

/* -------------------------------------- */
/* Generic convolution routines           */
/* -------------------------------------- */






/* -------------------------------------- */
/* Torch nxn wrappers                     */
/* -------------------------------------- */


static int nxn_(ConvProto_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);

  int stridex = luaT_getfieldcheckint(L, 1, "dW");
  int stridey = luaT_getfieldcheckint(L, 1, "dH");

  int padleft = luaT_getfieldcheckint(L, 1, "padleft");
  int padright = luaT_getfieldcheckint(L, 1, "padright");
  int padtop = luaT_getfieldcheckint(L, 1, "padtop");
  int padbottom = luaT_getfieldcheckint(L, 1, "padbottom");

  int overlap = luaT_getfieldcheckint(L, 1, "overlap");

  real alpha = luaT_getfieldchecknumber(L, 1, "alpha");
  real beta = luaT_getfieldchecknumber(L, 1, "beta");

  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *bias = luaT_getfieldcheckudata(L, 1, "bias", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);


#if 0
  int dimw = 2;
  int dimh = 1;
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");
#endif

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
  THTensor* icopy = THTensor_(newWithSize)(icopysize, NULL);

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

  /* copy kernel into kernel buffer */
  /* for now let's assert kernel is contiguous so we have to do nothing */

  THTensor* kcopy = weight;


  THTensor* ocopy = THTensor_(newWithSize4d)(bs, toh, tow, op);

  THTensor_(fill)(ocopy, 0);

   /* call GEMM */
	int hcall;
   for (hcall=0; hcall<nxs; hcall++) {
	int vcall;
      for (vcall=0; vcall<kh; vcall++) {
         int sq = vcall / stridey;
         int sr = vcall - sq * stridey;
         /* local icopy =  newSameTensor(input, stridey, bs, toh, tiw, ip) */
         /* float* iptr = torch.data(icopy[{sr+1,{},sq+1,hcall*stridex+1,{}}]) */
		 real* iptr = THTensor_(data)(icopy);
		 iptr       += (sr)*icopy->stride[0] + (sq)*icopy->stride[2] +  (hcall*stridex)*icopy->stride[3];

         /* local kptr  = torch.data(kcopy:select(1,vcall+1)) */
		 real* kptr = THTensor_(data)(kcopy);
		 kptr	 	+= vcall * kcopy->stride[0];

         /* local optr = torch.data(ocopy:select(3,hcall+1)) */
		 real* optr = THTensor_(data)(ocopy);
		 optr		+= hcall * ocopy->stride[2];


         int nrun = (bs-1)*toh*tow + oh*tow;
         int ngem = (nrun - hcall) / nxs;
         THBlas_(gemm)('T','N', op, ngem, kw*ip, 
              1, kptr, kw*ip, iptr, nxs*stridex*ip,
              1, optr, nxs*op ); 
      }
   }

  THTensor_(resize4d)(output, bs, oh, ow, op);

  real* ocpyptr = THTensor_(data)(ocopy);
  real* optr = THTensor_(data)(output);
  real* bptr = THTensor_(data)(bias);

  /* here we take alpha = 1 and beta = 0 */
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


  THTensor_(free)(icopy);
  THTensor_(free)(ocopy);

  /* luaL_error(L, "not implemented"); */
  return 0;
}


static int nxn_(ConvProto_updateGradInput)(lua_State *L)
{
#if 0
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  THTensor *tweight;
  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );
#endif

  luaL_error(L, "not implemented");
  return 0;
}


static int nxn_(ConvProto_accGradParameters)(lua_State *L)
{
#if 0
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);
  int dimw = 2;
  int dimh = 1;
  THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );
#endif

  luaL_error(L, "not implemented");
  return 0;
}

static const struct luaL_Reg nxn_(ConvProto__) [] = {
  {"ConvProto_updateOutput", nxn_(ConvProto_updateOutput)},
  {"ConvProto_updateGradInput", nxn_(ConvProto_updateGradInput)},
  {"ConvProto_accGradParameters", nxn_(ConvProto_accGradParameters)},
  {NULL, NULL}
};

static void nxn_(ConvProto_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nxn_(ConvProto__), "nxn");
  lua_pop(L,1);
}

#endif
