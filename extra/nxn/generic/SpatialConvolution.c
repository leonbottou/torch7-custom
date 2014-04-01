#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolution.c"
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


static int nxn_(SpatialConvolution_updateOutput)(lua_State *L)
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
  THTensor_(fill)(icopy, 0);

int s;
	#pragma omp parallel for private(s)
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
	/*#pragma omp parallel for private(hcall)*/
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

  THTensor_(fill)(output, 0);

  /* here we take alpha = 1 and beta = 0 */
	int itout0=0;
	int itocpy0=0;
	int it1;
	#pragma omp parallel for private(it1)
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


static int nxn_(SpatialConvolution_updateGradInput)(lua_State *L)
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

  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);

  int stridex = luaT_getfieldcheckint(L, 1, "dW");
  int stridey = luaT_getfieldcheckint(L, 1, "dH");

  int padleft = luaT_getfieldcheckint(L, 1, "padleft");
  int padright = luaT_getfieldcheckint(L, 1, "padright");
  int padtop = luaT_getfieldcheckint(L, 1, "padtop");
  int padbottom = luaT_getfieldcheckint(L, 1, "padbottom");

  int overlap = luaT_getfieldcheckint(L, 1, "overlap");
  //assert(overlap==1);

  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *revk;
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);


  int bs = input->size[0];
  int ih = input->size[1];
  int iw = input->size[2];
  int ip = input->size[3];

  int kh = weight->size[0];
  int op = weight->size[1];
  int kw = weight->size[2];
  assert(ip==weight->size[3]);


   assert(gradOutput->nDimension == 4)
   assert(bs == gradOutput->size[0])
   /* check that output h,w sizes match gradOutput sizes      */
   int goh = gradOutput->size[1];
   int gow = gradOutput->size[2];
   assert(goh == (ih + padtop + padbottom - kh) / stridey + 1) ;
   assert(gow == (iw + padleft + padright - kw) / stridex + 1) ;
   assert(op == gradOutput->size[3]);



   /* copyKernelReverse */
   int ko = weight->size[1];
   int ki = weight->size[3];
   
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

   revk = THTensor_(newWithSize)(revksize, NULL);
   THTensor_(fill)(revk, 0);
   
   real* koptr = THTensor_(data)(revk);
   real* kp    = THTensor_(data)(weight);
   
   int i=0;
   
   int stry, strx, ith, ycoord, ito, itw, xcoord, iti;

   int sh=weight->stride[0];
   int so=weight->stride[1];
   int sw=weight->stride[2];
   int si=weight->stride[3];

   for (stry=0; stry<stridey; stry++) {
		for (strx=0; strx<stridex; strx++) {
			for (ith=0; ith<kouth; ith++) {
				ycoord=kh-((ith)*stridey+stry+1);
				if (ycoord<kh && ycoord>-1) {
					for (ito=0; ito<kouto; ito++ ) {
            		for (itw=0; itw<koutw; itw++ ) {
							xcoord=kw-((itw)*stridex+strx+1);
							if (xcoord<kw && xcoord>-1) {
                  		for (iti=0; iti<kouti; iti++) {
									koptr[i] = kp[(ycoord)*sh+(iti)*so+(xcoord)*sw+(ito)*si];
									i=i+1;
                  		}
					 		}
               		else {
                  		i = i + kouti;
               		}
            		}
         		}
				}
         	else {
            		i = i + kouti*koutw*kouto;
         	}
			}
		}
	}


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

   THTensor * gradin = THTensor_(newWithSize)(gradinsize, NULL);
   THTensor_(fill)(gradin, 0);
   
   
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

   THTensor * gradOutCopy = THTensor_(newWithSize)(gradoutsize, NULL);
   THTensor_(fill)(gradOutCopy, 0);



   /*gradOutCopy = newSameTensor(gradOutput, bs+1, pgoh, pgow, op) 
   tgocopy=narrowTensorAndZero(gradOutCopy, 1, 1, bs)
   tgocopy=narrowTensorAndZero(tgocopy, 2, revkh, goh)
   tgocopy=narrowTensorAndZero(tgocopy, 3, revkw, gow)
   tgocopy:copy(gradOutput)*/


   real* goptr=THTensor_(data)(gradOutput);
   real* gocpyptr=THTensor_(data)(gradOutCopy);

   int itgocpy0=0;
   int itgo=0;

   int it1;
	/*#pragma omp parallel for private(it1)*/
   for (it1=0; it1<bs; it1++) {
      int  it2, it3, it4;
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



   /* GEMM calls : */
	/**/
	int nxs=1;
	if(!overlap) {nxs=revkw; /*printf("no overlap");*/}
	int hcall;
	/*#pragma omp parallel for private(hcall)*/
	for (hcall=0; hcall<nxs; hcall++) {
	   int stry, strx, vcall;
	   real* gradoutptr, *krevptr, *gradinptr;
	   int ldgradout, szkrev, ldkrev, ldgradin, nspots, ngem;
	   for (stry=0; stry<stridey; stry++) {
		   for (strx=0; strx<stridex; strx++) {
			   for (vcall=0; vcall<revkh; vcall++) {
               /*gradoutptr = torch.data(gradOutCopy[{1, revkh-vcall, 1, {}}])*/

				   gradoutptr = THTensor_(data)(gradOutCopy);
				   /*gradoutptr		+= (revkh-vcall-1)*gradOutCopy->stride[1];*/
				   gradoutptr		+= (revkh-vcall-1)*gradOutCopy->stride[1] + hcall*gradOutCopy->stride[2];
               /*int ldgradout    = op;*/
               ldgradout    = op*nxs;
                     
               /*krevptr    = torch.data(revk[{stry,strx,revkh-vcall,{},{},{}}])*/
				   krevptr	    = THTensor_(data)(revk);
				   krevptr 		+= (stry)*revk->stride[0] + (strx)*revk->stride[1] + (revkh-vcall-1)*revk->stride[2];
               szkrev       = op*revkw;
               ldkrev     	 = op*revkw;
                  
               /*gradinptr  = torch.data(gradin[{stry, 1, 1, stridex-(strx-1), {}}])*/
				   gradinptr	 = THTensor_(data)(gradin);
				   /*gradinptr		+= (stry)*gradin->stride[0] + (stridex-(strx)-1)*gradin->stride[3];*/
				   gradinptr		+= (stry)*gradin->stride[0] + (stridex-(strx)-1+hcall*stridex)*gradin->stride[3];
               /*int ldgradin   	 = ip * stridex;*/
               ldgradin   	 = ip * stridex * nxs;
                  
               nspots     = giw/stridex*gih*bs;
               ngem       = (nspots-hcall+nxs-1)/nxs;
                  
               /*THBlas_(gemm)('T', 'N', ip, nspots, szkrev, 1, krevptr, ldkrev, gradoutptr, ldgradout, 1, gradinptr, ldgradin);*/
               THBlas_(gemm)('T', 'N', ip, ngem, szkrev, 1, krevptr, ldkrev, gradoutptr, ldgradout, 1, gradinptr, ldgradin);           
			   }
		   }
	   }
   }


  /* correct padright and padbottom */
  int oldpadright = padright;
  int oldpadbottom = padbottom;
  padright = gow * stridex + kw - stridex - iw - padleft;
  padbottom = goh * stridey + kh - stridey - ih - padtop;
  /* assert(not exact or padright ~= oldpadright, "horizontal size mismatch"); */
  /* assert(not exact or padbottom ~= oldpadbottom, "horizontal size mismatch"); */
  if (padright < 0)  { padright = 0;}
  if (padbottom < 0) { padbottom = 0;}

  /* input size with padding */
  int piw = padleft + iw + padright; 
  int pih = padtop + ih + padbottom;



    
   int throwawayx=stridex - kw%stridex;
   int throwawayy=stridey - kh%stridey;
   if (stridex==1 || stridex==throwawayx) { throwawayx=0 ; } 
   if (stridey==1 || stridey==throwawayy) { throwawayy=0 ; }

   /* clean this after */ 
   int resw=piw;
   int resh=pih;

   THTensor * result = THTensor_(newWithSize4d)(bs, resh, resw, ip);
   THTensor_(fill)(result, 0);


   int itres0 = 0;
   int itgi0  = 0;
   int starty, sizey;

   real* gradinptr = THTensor_(data)(gradin);
   real* resptr = THTensor_(data)(result);

   for(stry=stridey; stry>0; stry--) 
   {
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
	
	   #pragma omp parallel for private(it1)
      for (it1=0; it1<bs; it1++) {
		   int itres1 = itres0 + it1*result->stride[0];
		   int itgi1  = itgi0  + it1*gradin->stride[1];
		   int it2, it3, it4;
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

THTensor_(resizeAs)(gradInput, input);
/*THTensor_(copy)(gradInput, result);*/

   real* gradinputptr = THTensor_(data)(gradInput);

   itgi0=0;
   itres0=0;
	#pragma omp parallel for private(it1)
	for (it1=0; it1<bs; it1++) {
		int itres1 = itres0 + it1*result->stride[0];
		int itgi1  = itgi0  + it1*gradInput->stride[0];
		int it2,it3,it4;
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
	} 


  THTensor_(free)(revk);
  THTensor_(free)(gradin);
  THTensor_(free)(result);
  THTensor_(free)(gradOutCopy);


   
/*
   for stry=stridey,1,-1 do   
       copy is tricky
      first line should be thrown away if 
      throwaway = stridey-stry < throwawayy
      
      tgicopy = gradin:select(1,stry)
      if throwaway then
         tgicopy=tgicopy:narrow(2, 2, gih-1)
      end
      tgicopy=tgicopy:narrow(3, 1+throwawayx, giw-throwawayx)
      
      
      -- select proper area in result tensor ()
      tresult = result:narrow(3,1, giw-throwawayx)
      if throwaway then
         tresult = tresult:narrow(2, (stridey-stry+1) - throwawayy + stridey, gih-1)
      else
         tresult = tresult:narrow(2, (stridey-stry+1) - throwawayy, gih)         
      end      
      
      local tresultSizes = tresult:size()
      local tresultStrides = tresult:stride()
      tresultStrides[2] = tresultStrides[2] * stridey
      tresult = tresult.new(tresult:storage(), tresult:storageOffset(), tresultSizes, tresultStrides)
      tresult:copy(tgicopy)
   end
 
   result=result:narrow(2, padtop+1, ih):narrow(3, padleft+1, iw)
   
   return result

*/










  /*luaL_error(L, "not implemented");*/
  return 0;
}


static int nxn_(SpatialConvolution_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  
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
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  THTensor *gradBias = luaT_getfieldcheckudata(L, 1, "gradBias", torch_Tensor);


#if 0
  int dimw = 2;
  int dimh = 1;
  luaL_argcheck(L, input->nDimension == 3 || input->nDimension == 4, 2, "3D or 4D(batch mode) tensor expected");
#endif

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
  THTensor_(fill)(icopy, 0);

  int s;
	#pragma omp parallel for private(s)
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

  THTensor* kcopy = gradWeight;


  THTensor* ocopy = THTensor_(newWithSize4d)(bs, toh, tow, op);
  THTensor_(fill)(ocopy, 0);
  
  real* gradoutptr=THTensor_(data)(gradOutput);
  real* ocpyptr = THTensor_(data)(ocopy);
  real* gradbiasptr = THTensor_(data)(gradBias);
  
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
               ocpyptr[itocpy3]  = gradoutptr[itout3];
               gradbiasptr[it4] += gradoutptr[itout3];
               itout3++;
               itocpy3++;
			   }
		   }
	   }
   }   
  
  THTensor_(mul)(gradBias, gradBias, scale);
  
  

   /* call GEMM */
	int hcall;
	/*#pragma omp parallel for private(hcall)*/
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
/*         THBlas_(gemm)('T','N', op, ngem, kw*ip, 
              1, kptr, kw*ip, iptr, nxs*stridex*ip,
              1, optr, nxs*op ); */
         THBlas_(gemm)('N','T', kw*ip,op, ngem, 
              scale, iptr, nxs*stridex*ip, optr, nxs*op, 
              1, kptr, kw*ip ); 
      }
   }


  THTensor_(free)(icopy);
  THTensor_(free)(ocopy);

  /* luaL_error(L, "not implemented"); */
  return 0;
  
  
}

static int nxn_(SpatialConvolution_clipWeights)(lua_State *L)
{
  real normbound = luaL_optnumber(L, 2, 1);
  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  
  int kh = weight->size[0];
  int op = weight->size[1];
  int kw = weight->size[2];
  int ip = weight->size[3];
  
  int str0 = weight->stride[0];
  int str1 = weight->stride[1];
  int str2 = weight->stride[2];
  int str3 = weight->stride[3];
  
  int ii1;
  real* wdata=THTensor_(data)(weight);
  
  #pragma omp parallel for private(ii1)
  for (ii1=0; ii1<op; ii1++)
  {
     int ii0,ii2,ii3;
     real filternorm=0;
     for (ii0=0; ii0<kh; ii0++)
     {
        for (ii2=0; ii2<kw; ii2++)
        {
           for (ii3=0; ii3<ip; ii3++)
           {
               filternorm+=wdata[ii0*str0+ii1*str1+ii2*str2+ii3*str3]*wdata[ii0*str0+ii1*str1+ii2*str2+ii3*str3];
           }
        }  
     }

     if (filternorm>normbound*normbound)
     {
         real scalefactor=normbound/sqrt(filternorm);
         for (ii0=0; ii0<kh; ii0++)
         {
            for (ii2=0; ii2<kw; ii2++)
            {
               for (ii3=0; ii3<ip; ii3++)
               {
                   wdata[ii0*str0+ii1*str1+ii2*str2+ii3*str3] *= scalefactor;
               }
            }  
         }
     }

  }
  
  return 1;
  
}


static const struct luaL_Reg nxn_(SpatialConvolution__) [] = {
  {"SpatialConvolution_updateOutput", nxn_(SpatialConvolution_updateOutput)},
  {"SpatialConvolution_updateGradInput", nxn_(SpatialConvolution_updateGradInput)},
  {"SpatialConvolution_accGradParameters", nxn_(SpatialConvolution_accGradParameters)},
  {"SpatialConvolution_clipWeights", nxn_(SpatialConvolution_clipWeights)},
  {NULL, NULL}
};

static void nxn_(SpatialConvolution_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nxn_(SpatialConvolution__), "nxn");
  lua_pop(L,1);
}

#endif

