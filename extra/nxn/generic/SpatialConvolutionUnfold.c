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
Strides in convolution : dW, dH. */ 



/* -------------------------------------- */
/* Generic functions                      */
/* -------------------------------------- */

/* fillRow : 
-	give the coordinates of the output pixel (y_out, x_out)
-	give the size of the kernel (kH, kW, nInputPlane) 
-	the strides (dW, dH)
-	the paddings (up and left only)
-	give the pointer to the row in kernelSlices
It will fill the row with the proper data to build the Toeplitz matrix. */ 

void nxn_(fillRow)(real* ksliceptr_row, real* inputptr, int batchidx, int y_out, int x_out, int kH, int kW, int dH, int dW, int isize1, int isize2, int nInputPlane, int inputstride0, int inputstride1, int inputstride2, int padup, int padleft)
{
	real* inputdata = inputptr + batchidx*inputstride0;
	int yslice, xslice, y_in, x_in;
	y_in=y_out*dH-padup;
	x_in=x_out*dW-padleft;
	for(yslice=0; yslice<kH; yslice++)
	{
		for(xslice=0; xslice<kW; xslice++)
		{
			real* kptrtmp = ksliceptr_row + yslice * (kW*nInputPlane) + xslice*nInputPlane;
			if(y_in+yslice < 0 || y_in+yslice >= isize1 || x_in+xslice < 0 || x_in+xslice >= isize2) 
			{
				memset(kptrtmp, 0, nInputPlane*sizeof(real));
			}
			else
			{
				real* iptrtmp = inputdata + (y_in+yslice) * inputstride1 + (x_in+xslice) * inputstride2;
				memcpy(kptrtmp, iptrtmp, nInputPlane*sizeof(real));
			}
		}
	}
}





/* addRow : 
-	give the coordinates of the output pixel (y_out, x_out)
-	give the size of the kernel (kH, kW, nInputPlane) 
-	the strides (dW, dH)
-	the paddings (up and left only)
-	give the pointer to the row in kernelSlices
It will add up the values of the row and sum them in the input tensor. (useful for gradInput) */ 

void nxn_(addRow)(real* ksliceptr_row, real* inputptr, int batchidx, int y_out, int x_out, int kH, int kW, int dH, int dW, int isize1, int isize2, int nInputPlane, int inputstride0, int inputstride1, int inputstride2, int padup, int padleft)
{
	real* inputdata = inputptr + batchidx*inputstride0;
	int yslice, xslice, y_in, x_in;
	y_in=y_out*dH-padup;
	x_in=x_out*dW-padleft;
	for(yslice=0; yslice<kH; yslice++)
	{
		for(xslice=0; xslice<kW; xslice++)
		{
			real* kptrtmp = ksliceptr_row + yslice * (kW*nInputPlane) + xslice*nInputPlane;
			if(!(y_in+yslice < 0 || y_in+yslice >= isize1 || x_in+xslice < 0 || x_in+xslice >= isize2)) 
			{
				real* iptrtmp = inputdata + (y_in+yslice) * inputstride1 + (x_in+xslice) * inputstride2;
				int it;
				for(it=0; it<nInputPlane; it++) 
				{
					#pragma omp atomic
					iptrtmp[it] += kptrtmp[it];
				}
			}
		}
	}
}




/* Toeplitz takes an input tensor and stores  
   stores the corresponding Toeplitz matrix in   
   the kernelSlices tensor, assuming it has the
   proper size. 
     - size of kernelSlices: batchsize * (width * height of output), kW * kW * inputPlane
     - batchsize has to be the same as in the Tensor input
     - but input can be narrowed over the first dimension

   We just loop over the input pixels and copy what they need.
   Whenever possible, we memcpy blocks of size kW*nInputPlane.

	If add==1, then we do the reverse operation, and add up the values of the Toeplitz matrix
	(useful to obtain gradInput). */

void nxn_(Toeplitz)(THTensor *input, THTensor* kernelSlices, int kH, int kW, int dH, int dW, int padup, int paddown, int padleft, int padright, int kslicerow_min, int kslicerow_max, int add)
{
	/* find the size of kernelslices */
	int batchsize = input->size[0];
	int isize1 = input->size[1];
	int isize2 = input->size[2];
	int nInputPlane = input->size[3];
	int size1 = (isize1 - kH + padup + paddown) / dH + 1;
	int size2 = (isize2 - kW + padleft + padright) / dW + 1;

	int inputstride0 = input->stride[0];
	int inputstride1 = input->stride[1];
	int inputstride2 = input->stride[2];

	real* inputptr = THTensor_(data)(input);

	int numrows=kslicerow_max-kslicerow_min;
	real* kslicedata=THTensor_(data)(kernelSlices);
	THTensor_(resize2d)(kernelSlices, numrows, kW*kH*nInputPlane);

	int rowidx;
	int y_out, x_out;
	int batchidx;
	#pragma omp parallel for private(rowidx)
	for (rowidx=0; rowidx<numrows; rowidx++)
	{
		batchidx = (kslicerow_min + rowidx) / (size1*size2);
		y_out=(kslicerow_min + rowidx - batchidx*(size1*size2)) / size2;
		x_out=(kslicerow_min + rowidx - batchidx*(size1*size2)) % size2;
		real* ksliceptr_row = kslicedata + rowidx * (kW*kH*nInputPlane);
		if(add) 
		{
			nxn_(addRow)(ksliceptr_row, inputptr, batchidx, y_out, x_out, kH, kW, dH, dW, isize1, isize2, nInputPlane, inputstride0, inputstride1, inputstride2, padup, padleft);
		}
		else
		{
			nxn_(fillRow)(ksliceptr_row, inputptr, batchidx, y_out, x_out, kH, kW, dH, dW, isize1, isize2, nInputPlane, inputstride0, inputstride1, inputstride2, padup, padleft);
		}
	}
}



/* sliceInput is the forward wrapper around Toeplitz */

inline void nxn_(sliceInput)(THTensor *input, THTensor* kernelSlices, int kH, int kW, int dH, int dW, int padup, int paddown, int padleft, int padright, int kslicerow_min, int kslicerow_max)
{
	nxn_(Toeplitz)(input, kernelSlices, kH, kW, dH, dW, padup, paddown, padleft, padright, kslicerow_min, kslicerow_max, 0);
}


/* unsliceGradient is the backward wrapper around Toeplitz */

inline void nxn_(unsliceGradient)(THTensor *input, THTensor* kernelSlices, int kH, int kW, int dH, int dW, int padup, int paddown, int padleft, int padright, int kslicerow_min, int kslicerow_max)
{
	nxn_(Toeplitz)(input, kernelSlices, kH, kW, dH, dW, padup, paddown, padleft, padright, kslicerow_min, kslicerow_max, 1);
}





/* -------------------------------------- */
/* Torch nxn wrappers                     */
/* -------------------------------------- */


static int nxn_(SpatialConvolutionUnfold_updateOutput)(lua_State *L)
{
	THTensor *input = luaT_checkudata(L, 2, torch_Tensor);

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
	THTensor_(resize2d)(output, batchsize* size1* size2, nOutputPlane);


	/* here we can set an upper limit to the number of rows of the unfolded input */
	/* however it is still unclear how to pick the proper limit */

	int totalNumRows=batchsize*size1*size2;
	int ompthr=omp_get_max_threads();
	int rowlimit = MIN(totalNumRows, nOutputPlane*ompthr); /* so the unfolded matrix split is of same shape as kernels */
	
	int numSplits=(totalNumRows*ompthr+rowlimit-1)/rowlimit;
	int numRowsInSplit=(totalNumRows+numSplits-1)/numSplits;

	int split;
	#pragma omp parallel for private(split)
	for(split=0; split<numSplits; split++)
	{
		int splitSize=numRowsInSplit;
		if(split*numRowsInSplit+splitSize > totalNumRows)
		{
			splitSize=totalNumRows-split*numRowsInSplit;
		}
		THTensor* kSlicesSplit = THTensor_(newWithSize2d)(splitSize, kW*kH*nInputPlane);

		int kslicerow_min = split*numRowsInSplit;
		int kslicerow_max = split*numRowsInSplit + splitSize;

		THTensor* outputSplit = THTensor_(newNarrow)(output, 0, kslicerow_min, splitSize);
		nxn_(sliceInput)(input, kSlicesSplit, kH, kW, dH, dW, padup, paddown, padleft, padright, kslicerow_min, kslicerow_max);
		THTensor_(addmm)(outputSplit, 0, outputSplit, 1, kSlicesSplit, kernels);
	
		THTensor_(free)(kSlicesSplit);
	}
	THTensor_(resize4d)(output, batchsize, size1, size2, nOutputPlane);

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
	THTensor_(fill)(gradInput, 0);

	THTensor *kernels = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);

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

	/* here we can set an upper limit to the number of rows of the unfolded gradInput */
	/* however it is still unclear how to pick the proper limit */

	int totalNumRows=batchsize*size1*size2;
	int ompthr=omp_get_max_threads();
	int rowlimit = MIN(totalNumRows, kW*kH*nInputPlane*ompthr); /* so the unfolded matrix split is of same shape as kernels */
	
	int numSplits=(totalNumRows+rowlimit-1)/rowlimit;
	numSplits *= ompthr;
	int numRowsInSplit=(totalNumRows+numSplits-1)/numSplits;

	THTensor_(resize2d)(gradOutput,totalNumRows,nOutputPlane);
	THTensor_(resize2d)(kernels, nOutputPlane, kW*kH*nInputPlane);

	int split;
	#pragma omp parallel for private(split)
	for(split=0; split<numSplits; split++)
	{
		int splitSize=numRowsInSplit;
		if(split*numRowsInSplit+splitSize > totalNumRows)
		{
			splitSize=totalNumRows-split*numRowsInSplit;
		}
		THTensor* kSlicesSplit = THTensor_(newWithSize2d)(splitSize, kW*kH*nInputPlane);

		int kslicerow_min = split*numRowsInSplit;
		int kslicerow_max = split*numRowsInSplit + splitSize;

		THTensor* gradOutputSplit = THTensor_(newNarrow)(gradOutput, 0, kslicerow_min, splitSize);
      THTensor_(addmm)(kSlicesSplit, 0, kSlicesSplit, 1, gradOutputSplit, kernels);
		nxn_(unsliceGradient)(gradInput, kSlicesSplit, kH, kW, dH, dW, padup, paddown, padleft, padright, kslicerow_min, kslicerow_max);
		THTensor_(free)(kSlicesSplit);
	}

	THTensor_(resize4d)(kernels, nOutputPlane, kH, kW, nInputPlane);

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
