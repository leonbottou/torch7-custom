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

/* We implement here a convolution algorithm that unfolds a matrix into 
something close to a Toeplitz matrix ("kernelSlices") in a way that if 
you multiply it by a matrix of flattened filters you do a convolution : 
- each row contains one cell (ie. the sub-image that is dot-producted with 
a convolution filter to obtain a single output value).
- each column of the weight matrix contains a flattened weight tensor.

result = kernelSlices * weights^T

Then the result is resized to obtain a 4D tensor as output.
Input can be zero-padded (only if you provide the values).
Strides in convolution : dW, dH.

*/ 

/* -------------------------------------- */
/* Generic functions                      */
/* -------------------------------------- */

/* sliceInput takes an input tensor and stores  
   stores the corresponding Toeplitz matrix in   
   the kernelSlices tensor, assuming it has the
   proper size. 
     - size of kernelSlices: batchsize * (width * height of output), kW * kW * inputPlane
     - batchsize has to be the same as in the Tensor input
     - but input can be narrowed over the first dimension

   We just loop over the input pixels and copy what they need.
   Whenever possible, we memcpy blocks of size kW*nInputPlane.

*/
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

/* unsliceGradient does the same as sliceInput, except 
   it sums up the values from the Toeplitz matrix to obtain
   the gradInput tensor.

   It takes as input a Toeplitz matrix backwardSlices and a
   gradInput tensor of same size as the input.

*/


void nxn_(unsliceGradient)(THTensor *backwardSlices, THTensor *gradInput, int kH, int kW, int dH, int dW, int padup, int paddown, int padleft, int padright)
{
	/* find the size of kernelslices */
	int batchsize = gradInput->size[0];
	int isize1 = gradInput->size[1];
	int isize2 = gradInput->size[2];
	int nInputPlane = gradInput->size[3];
	int size1 = (isize1 - kH + padup + paddown) / dH + 1;
	int size2 = (isize2 - kW + padleft + padright) / dW + 1;

	THTensor_(fill)(gradInput, 0);
	THTensor_(resize2d)(backwardSlices, batchsize*size1*size2, kW*kH*nInputPlane);

	int batchidx;
#pragma omp parallel for private(batchidx)
	for (batchidx=0; batchidx<batchsize; batchidx++)
	{
		real* gradInputdata=THTensor_(data)(gradInput) + batchidx*gradInput->stride[0];
		real* bslicedata=THTensor_(data)(backwardSlices) + batchidx*size1*size2*kW*kH*nInputPlane;

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
					for(xslice=0; xslice<kW; xslice++)
					{
						if(x_in+xslice < 0 || x_in+xslice >= isize2) continue;
						real* bptrtmp = bslicedata + (y_out*size2+x_out) * (kW*kH*nInputPlane) + yslice * (kW*nInputPlane) + xslice*nInputPlane;
						real* gptrtmp = gradInputdata + (y_in+yslice) * gradInput->stride[1] + (x_in+xslice) * gradInput->stride[2];
						int it;
						for (it=0; it<nInputPlane; it++)
						{
							gptrtmp[it] += bptrtmp[it];
						}
					}
				}
			}
		}
	}

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

	return 0;
}


static int nxn_(SpatialConvolutionUnfold_updateGradInput)(lua_State *L)
{

	THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
	THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
	THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
	THTensor_(resizeAs)(gradInput, input);
	THTensor *kernels = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
	int dW = luaT_getfieldcheckint(L, 1, "dW");
	int dH = luaT_getfieldcheckint(L, 1, "dH");

	int kW = luaT_getfieldcheckint(L, 1, "kW");
	int kH = luaT_getfieldcheckint(L, 1, "kH");
	int nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
	THArgCheck( nOutputPlane == gradOutput->size[input->nDimension == 4 ? 1 : 0], 1, "Number of output features is not equal to nOutputPlane" );
	
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
	
	THTensor* backwardSlices = THTensor_(newWithSize3d)(batchsize,size1*size2,kW*kH*nInputPlane);

        int nsplits = luaT_getfieldcheckint(L, 1, "nsplits");
	
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
                THTensor* bSliceSplit = THTensor_(newNarrow)(backwardSlices, 0, split*newbatchsize, splitsize);
		THTensor_(resize2d)(bSliceSplit, splitsize*size1*size2, kW*kH*nInputPlane);
                THTensor* ginputSplit  = THTensor_(newNarrow)(gradInput, 0, split*newbatchsize, splitsize);
                THTensor* goutputsplit = THTensor_(newNarrow)(gradOutput, 0, split*newbatchsize, splitsize);
                THTensor_(resize2d)(goutputsplit, splitsize* size1* size2, nOutputPlane);

                THTensor_(addmm)(bSliceSplit, 0, bSliceSplit, 1, goutputsplit, kernels);
                nxn_(unsliceGradient)(bSliceSplit, ginputSplit, kH, kW, dH, dW, padup, paddown, padleft, padright);

        }

	THTensor_(free)(backwardSlices);
	return 0;
}


static int nxn_(SpatialConvolutionUnfold_accGradParameters)(lua_State *L)
{
/*	THTensor *input = luaT_checkudata(L, 2, torch_Tensor);
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

