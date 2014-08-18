#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialConvolutionUnfold.c"
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

void nxn_(sliceInput)(THTensor *input, THTensor* kernelSlices, int kH, int kW, int dH, int dW, int padup, int paddown, int padleft, int padright)
{
   /* find the size of kernelslices */
	int batchsize = input->size[0];
	int isize1 = input->size[1];
	int isize2 = input->size[2];
	int nInputPlane = input->size[3];
	int size1 = (isize1 - kH + padup + paddown) / dH + 1;
	int size2 = (isize2 - kW + padleft + padright) / dW + 1;

	THTensor_(resize2d)(kernelSlices, batchsize*size1*size2, kW*kH*nInputPlane);
	THTensor_(fill)(kernelSlices, 0);

	int batchidx;
	#pragma omp parallel for private(batchidx)
	for (batchidx=0; batchidx<batchsize; batchidx++)
	{
		real* inputdata=THTensor_(data)(input) + batchidx*input->stride[0];
		real* kslicedata=THTensor_(data)(kernelSlices) + batchidx*size1*size2*kW*kH*nInputPlane;

		int y_out, x_out, y_in, x_in, yslice, xslice;
		for (y_out=0; y_out<size1; y_out++)
		{
			y_in=y_out*dH-padup;
			for(yslice=0; yslice<kH; yslice++)
			{
				if(y_in+yslice < 0 || y_in+yslice >= isize1) continue;
				for (x_out=0; x_out<size2; x_out++)
				{
					x_in=x_out*dW-padleft;
					if(x_in >= 0 && x_in+kW < isize2)
					{
						real* kptrtmp = kslicedata + (y_out*size2+x_out) * (kW*kH*nInputPlane) + yslice * (kW*nInputPlane);
						real* iptrtmp = inputdata + (y_in+yslice) * input->stride[1] + (x_in) * input->stride[2];
						memcpy(kptrtmp, iptrtmp, kW*nInputPlane*sizeof(real));
					}
					else
					{
						for(xslice=0; xslice<kW; xslice++)
						{
							if(x_in+xslice < 0 || x_in+xslice >= isize2) continue;
							real* kptrtmp = kslicedata + (y_out*size2+x_out) * (kW*kH*nInputPlane) + yslice * (kW*nInputPlane) + xslice*nInputPlane;
							real* iptrtmp = inputdata + (y_in+yslice) * input->stride[1] + (x_in+xslice) * input->stride[2];
							memcpy(kptrtmp, iptrtmp, nInputPlane*sizeof(real));
						}
					}
				}
			}
		}
	}
}

void nxn_(unsliceGradient)(THTensor *backwardSlices, THTensor *gradInput, THTensor *gradOutput, int kH, int kW, int dH, int dW, int padup, int paddown, int padleft, int padright)
{

}

/* -------------------------------------- */
/* Torch nxn wrappers                     */
/* -------------------------------------- */


static int nxn_(SpatialConvolutionUnfold_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);

  int nsplits = luaT_getfieldcheckint(L, 1, "nsplits");

  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");
  int kW = luaT_getfieldcheckint(L, 1, "kW");
  int kH = luaT_getfieldcheckint(L, 1, "kH");
  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");

  int padleft = luaT_getfieldcheckint(L, 1, "padleft");
  int padright = luaT_getfieldcheckint(L, 1, "padright");
  int padup = luaT_getfieldcheckint(L, 1, "padtop");
  int paddown = luaT_getfieldcheckint(L, 1, "padbottom");

  int batchsize=input->size[0];
  int isize1 = input->size[1];
  int isize2 = input->size[2];
  int nInputPlane = input->size[3];
  int size1 = (isize1 - kH + padup + paddown) / dH + 1;
  int size2 = (isize2 - kW + padleft + padright) / dW + 1;


  THTensor *kernels = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor_(resize2d)(kernels, nOutputPlane, kW*kH*nInputPlane);
  THTensor_(transpose)(kernels, NULL, 0, 1);

  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);
  THTensor_(resize4d)(output, batchsize, size1, size2, nOutputPlane);

  THTensor *kernelSlices = luaT_checkudata(L, 3, torch_Tensor);
  THTensor_(resize3d)(kernelSlices, batchsize, size1*size2, kW*kH*nInputPlane);


   int newbatchsize=(batchsize+nsplits-1)/nsplits;
	int split;
	#pragma omp parallel for private(split)
	for(split=0; split<nsplits; split++)
	{
		int splitsize=newbatchsize;
		if(split*newbatchsize+splitsize > batchsize)
		{
			splitsize=batchsize-split*newbatchsize;
		}
		THTensor* kSliceSplit = THTensor_(newNarrow)(kernelSlices, 0, split*newbatchsize, splitsize);
      THTensor* inputSplit  = THTensor_(newNarrow)(input, 0, split*newbatchsize, splitsize);
      THTensor* outputsplit = THTensor_(newNarrow)(output, 0, split*newbatchsize, splitsize);
   	THTensor_(resize2d)(outputsplit, splitsize* size1* size2, nOutputPlane);

	   nxn_(sliceInput)(inputSplit, kSliceSplit, kH, kW, dH, dW, padup, paddown, padleft, padright);
  		THTensor_(addmm)(outputsplit, 1, outputsplit, 1, kSliceSplit, kernels);

	}


  THTensor_(transpose)(kernels, NULL, 0, 1);
  THTensor_(resize4d)(kernels, nOutputPlane, kH, kW, nInputPlane);

  /* luaL_error(L, "not implemented"); */
  return 0;
}


static int nxn_(SpatialConvolutionUnfold_updateGradInput)(lua_State *L)
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


  /*luaL_error(L, "not implemented");*/
  return 0;
}


static int nxn_(SpatialConvolutionUnfold_accGradParameters)(lua_State *L)
{
/*  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);
  
  int stridex = luaT_getfieldcheckint(L, 1, "dW");
  int stridey = luaT_getfieldcheckint(L, 1, "dH");

  int padleft = luaT_getfieldcheckint(L, 1, "padleft");
  int padright = luaT_getfieldcheckint(L, 1, "padright");
  int padtop = luaT_getfieldcheckint(L, 1, "padtop");
  int padbottom = luaT_getfieldcheckint(L, 1, "padbottom");

  int overlap = luaT_getfieldcheckint(L, 1, "overlap");

  int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "tmpgradweight", torch_Tensor);
  THTensor *tmpgradweight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  //printf("transposing");
  THTensor_(transpose)(gradWeight,tmpgradweight, 0, 1);
  gradWeight = THTensor_(newContiguous)(gradWeight);
  //printf("transposing done");*/


  /* luaL_error(L, "not implemented"); */
  return 0;
  
  
}


static const struct luaL_Reg nxn_(SpatialConvolutionUnfold__) [] = {
  {"SpatialConvolutionUnfold_updateOutput", nxn_(SpatialConvolutionUnfold_updateOutput)},
  {"SpatialConvolutionUnfold_updateGradInput", nxn_(SpatialConvolutionUnfold_updateGradInput)},
  {"SpatialConvolutionUnfold_accGradParameters", nxn_(SpatialConvolutionUnfold_accGradParameters)},
  {NULL, NULL}
};

static void nxn_(SpatialConvolutionUnfold_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nxn_(SpatialConvolutionUnfold__), "nxn");
  lua_pop(L,1);
}

#endif

