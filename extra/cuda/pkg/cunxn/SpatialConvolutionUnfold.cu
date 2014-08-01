#ifndef assert
#define assert(e)  \
    if (!(e)) { \
        printf("failed assertion `%s'\n", #e); \
        THError("aborting..."); \
    };
#endif

/*

This file contains 4 kernels :
- copyPixelsInSlices and its more optimized version copyPixelsInSlicesReg (when there is an upper bound on the number of planes).
- addPixelsInSlices and its optimized version addPixelsInSlicesReg.

The primary kernel is copyPixelsInSlices : it unfolds a 3D matrix into a 2D matrix in a way that the 2D convolution (with many kernels) becomes a matrix multiplication.
We call the resulting matrix "kernelSlices". Each row corresponds to a kW*kH*nInputPlane array.

Steps :
1) choose a pixel (pixi = blockIdx.x, pixj = blockIdx.y)
2) find which slices (coordinates (imin-imax, jmin-jmax)) will contain the pixel information
3) loop : copy the pixel information, jump to next slice (and position) by 
		moving the kernelSlices pointer ptrkslices by stridej = (kH*kW - dW) * nInputPlane

	detailed example : pixel (4,4), kernels of size 5*5, stride dW=1 :
	- 1st slice  : top-left coordinates : (imin,jmin)  . Pixel is in coordinates (4,4, position 25) of the slice.
	- 2nd slice  : top-left coordinates : (imin,jmin+1). Pixel is in coordinates (4,3, position 24) of the slice.
	- 3rd slice  : top-left coordinates : (imin,jmin+2). Pixel is in coordinates (4,2, position 23) of the slice.
	- 4th slice  : top-left coordinates : (imin,jmin+2). Pixel is in coordinates (4,1, position 22) of the slice.
	- 5th slice  : top-left coordinates : (imin,jmin+2). Pixel is in coordinates (4,0, position 21) of the slice.
	- when jmax-jmin slices have been filled, we jump to the next series of slices by 
		moving ptrkslices by stridei = (((size2-jmax+jmin-1)*kH -dH)*kW  + (jmax-jmin+1)*dW)*nInputPlane
	- 1st slice  : top-left coordinates : (imin+1,jmin)  . Pixel is in coordinates (3,4, position 20) of the slice.
	- 2nd slice  : top-left coordinates : (imin+1,jmin+1). Pixel is in coordinates (3,3, position 19) of the slice.
	- 3rd slice  : top-left coordinates : (imin+1,jmin+2). Pixel is in coordinates (3,2, position 18) of the slice.
	- 4th slice  : top-left coordinates : (imin+1,jmin+2). Pixel is in coordinates (3,1, position 17) of the slice.
	- 5th slice  : top-left coordinates : (imin+1,jmin+2). Pixel is in coordinates (3,0, position 16) of the slice.
	- ...

In case the pixel (pixi,pixj) is in the zero-padding, we fill the slice with zeros.

addPixelsInSlices is the same, except we read the contents of the array instead of writing.

the *Reg versions just consist in preloading the pixel information before writing it.


*/

__global__ void copyPixelsInSlices(float *ptrinput, float *ptrkslices,
	int dH, int dW, int kH, int kW, int size1, int size2, int isize1, int isize2, int nInputPlane, int valuesperthread, int padleft, int padright, int padup, int paddown, int inputstr0, int kslicesstr0)
{
	const int pixi=blockIdx.x;
	const int pixj=blockIdx.y;
	const int blk =blockDim.x;
	const int tidx=threadIdx.x;

        int imin=(pixi - (kH - 1) + (dH -1))/dH > 0 ? (pixi - (kH - 1) + (dH -1))/dH : 0 ;
        int jmin=(pixj - (kW - 1) + (dW -1))/dW > 0 ? (pixj - (kW - 1) + (dW -1))/dW : 0 ;
        int imax= pixi / dH < size1 ? pixi / dH : size1 - 1 ;
        int jmax= pixj / dW < size2 ? pixj / dW : size2 - 1 ;

	int i;
	int j;
	int k;

	bool zeropad=pixi<padup || pixi>isize1-1+padup || pixj<padleft || pixj>isize2-1+padleft ;
	
	ptrinput   += inputstr0*blockIdx.z + ((pixi-padup) * isize2 + (pixj-padleft)) * nInputPlane ;
	ptrkslices += kslicesstr0*blockIdx.z + ((imin * size2  + jmin) * kH * kW +  (pixi - imin * dH) * kW + (pixj - jmin*dW) ) * nInputPlane;

	int stridej = (kH*kW - dW) * nInputPlane;
	int stridei = (((size2-jmax+jmin-1)*kH -dH)*kW  + (jmax-jmin+1)*dW)*nInputPlane;
	
		for(i=imin; i<imax+1; i++) {
			for(j=jmin; j<jmax+1; j++) {
				if(zeropad) 
				{
					for(k=threadIdx.x; k<nInputPlane; k+=blockDim.x) {
						ptrkslices[k]=0;
					}
				}
				else {
					for(k=threadIdx.x; k<nInputPlane; k+=blockDim.x) {
						ptrkslices[k]=ptrinput[k];
					}
				}
				ptrkslices += stridej;
			}
			ptrkslices += stridei;
		}	

}


/*__global__ void addPixelsInSlices(float *ptrgradinput, float *ptrkslices,
	int dH, int dW, int kH, int kW, int size1, int size2, int isize1, int isize2, int nInputPlane, int valuesperthread, int padleft, int padright, int padup, int paddown)
{
	const int pixi=blockIdx.x;
	const int pixj=blockIdx.y;
	const int blk =blockDim.x;
	const int tidx=threadIdx.x;

        int imin=(pixi - (kH - 1) + (dH -1))/dH > 0 ? (pixi - (kH - 1) + (dH -1))/dH : 0 ;
        int jmin=(pixj - (kW - 1) + (dW -1))/dW > 0 ? (pixj - (kW - 1) + (dW -1))/dW : 0 ;
        int imax= pixi / dH < size1 ? pixi / dH : size1 - 1 ;
        int jmax= pixj / dW < size2 ? pixj / dW : size2 - 1 ;

	int i;
	int j;
	int k;

	bool zeropad=pixi<padup || pixi>isize1-1+padup || pixj<padleft || pixj>isize2-1+padleft ;
	
	ptrgradinput += ((pixi-padup) * isize2 + (pixj-padleft)) * nInputPlane ;
	ptrkslices   += ((imin * size2  + jmin) * kH * kW +  (pixi - imin * dH) * kW + (pixj - jmin*dW) ) * nInputPlane;

	int stridej = (kH*kW - dW) * nInputPlane;
	int stridei = (((size2-jmax+jmin-1)*kH -dH)*kW  + (jmax-jmin+1)*dW)*nInputPlane;

	for(k=0; k<valuesperthread; k++) {
		ptrgradinput[k*blk+tidx] = 0;
	}
	
	if(tidx<nInputPlane) {
		if(!zeropad) {
			for(i=imin; i<imax+1; i++) {
				for(j=jmin; j<jmax+1; j++) {
						for(k=0; k<valuesperthread; k++) {
							ptrgradinput[k*blk+tidx] += ptrkslices[k*blk+tidx];
						}
					ptrkslices += stridej;
				}
				ptrkslices += stridei;
			}	
		}
	}
}*/


__global__ void addPixelsInSlices(float *ptrgradinput, float *ptrkslices,
	int dH, int dW, int kH, int kW, int size1, int size2, int isize1, int isize2, int nInputPlane, int padleft, int padright, int padup, int paddown, int gradinputstr0, int kslicesstr0)
{
	const int pixi=blockIdx.x;
	const int pixj=blockIdx.y;
	const int blk =blockDim.x;
	const int tidx=threadIdx.x;

        int imin=(pixi - (kH - 1) + (dH -1))/dH > 0 ? (pixi - (kH - 1) + (dH -1))/dH : 0 ;
        int jmin=(pixj - (kW - 1) + (dW -1))/dW > 0 ? (pixj - (kW - 1) + (dW -1))/dW : 0 ;
        int imax= pixi / dH < size1 ? pixi / dH : size1 - 1 ;
        int jmax= pixj / dW < size2 ? pixj / dW : size2 - 1 ;

	int i;
	int j;
	int k;

	bool zeropad=pixi<padup || pixi>isize1-1+padup || pixj<padleft || pixj>isize2-1+padleft ;
	
	ptrgradinput += gradinputstr0*blockIdx.z + ((pixi-padup) * isize2 + (pixj-padleft)) * nInputPlane ;
	ptrkslices += kslicesstr0*blockIdx.z + ((imin * size2  + jmin) * kH * kW +  (pixi - imin * dH) * kW + (pixj - jmin*dW) ) * nInputPlane;

	int stridej = (kH*kW - dW) * nInputPlane;
	int stridei = (((size2-jmax+jmin-1)*kH -dH)*kW  + (jmax-jmin+1)*dW)*nInputPlane;
	
		for(i=imin; i<imax+1; i++) {
			for(j=jmin; j<jmax+1; j++) {
				if(zeropad) 
				{
					
				}
				else {
					for(k=threadIdx.x; k<nInputPlane; k+=blockDim.x) {
						ptrgradinput[k] += ptrkslices[k];
					}
				}
				ptrkslices += stridej;
			}
			ptrkslices += stridei;
		}	

}




template <int maxnumplanes> __global__ void addPixelsInSlicesReg(float *ptrgradinput, float *ptrkslices,
	int dH, int dW, int kH, int kW, int size1, int size2, int isize1, int isize2, int nInputPlane, int padleft, int padright, int padup, int paddown, int gradinputstr0, int kslicesstr0)
{
	const int pixi=blockIdx.x;
	const int pixj=blockIdx.y;
	const int blk =blockDim.x;
	const int tidx=threadIdx.x;

        int imin=(pixi - (kH - 1) + (dH -1))/dH > 0 ? (pixi - (kH - 1) + (dH -1))/dH : 0 ;
        int jmin=(pixj - (kW - 1) + (dW -1))/dW > 0 ? (pixj - (kW - 1) + (dW -1))/dW : 0 ;
        int imax= pixi / dH < size1 ? pixi / dH : size1 - 1 ;
        int jmax= pixj / dW < size2 ? pixj / dW : size2 - 1 ;

	int i;
	int j;
	int k;

	float gradvalues[maxnumplanes/32];
		for(k=0; k<maxnumplanes/32; k++) {
			gradvalues[k]=0;
		}

	bool zeropad=pixi<padup || pixi>isize1-1+padup || pixj<padleft || pixj>isize2-1+padleft ;
	
	ptrgradinput += gradinputstr0*blockIdx.z + ((pixi-padup) * isize2 + (pixj-padleft)) * nInputPlane ;
	ptrkslices += kslicesstr0*blockIdx.z + ((imin * size2  + jmin) * kH * kW +  (pixi - imin * dH) * kW + (pixj - jmin*dW) ) * nInputPlane;

	int stridej = (kH*kW - dW) * nInputPlane;
	int stridei = (((size2-jmax+jmin-1)*kH -dH)*kW  + (jmax-jmin+1)*dW)*nInputPlane;
	
	if(!zeropad) 
		{
			for(i=imin; i<imax+1; i++) {
				for(j=jmin; j<jmax+1; j++) {
					
	/*					for(k=threadIdx.x; k<nInputPlane; k+=blockDim.x) {
							ptrgradinput[k] += ptrkslices[k];
						}*/
						for(k=0; k*blockDim.x+threadIdx.x<nInputPlane; k++) {
							gradvalues[k] += ptrkslices[k*blockDim.x+threadIdx.x];
					}
					ptrkslices += stridej;
				}
				ptrkslices += stridei;
			}
	
			for(k=0; k*blockDim.x+threadIdx.x<nInputPlane; k++) {
				ptrgradinput[k*blockDim.x+threadIdx.x] = gradvalues[k];
			}
		}


}


/*template <int maxnumplanes> __global__ void addPixelsInSlicesReg(float *ptrgradinput, float *ptrkslices,
	int dH, int dW, int kH, int kW, int size1, int size2, int isize1, int isize2, int nInputPlane, int valuesperthread, int padleft, int padright, int padup, int paddown, int gradinputstr0, int kslicesstr0)
{
	const int pixi=blockIdx.x;
	const int pixj=blockIdx.y;
	const int blk =blockDim.x;
	const int tidx=threadIdx.x;

        int imin=(pixi - (kH - 1) + (dH -1))/dH > 0 ? (pixi - (kH - 1) + (dH -1))/dH : 0 ;
        int jmin=(pixj - (kW - 1) + (dW -1))/dW > 0 ? (pixj - (kW - 1) + (dW -1))/dW : 0 ;
        int imax= pixi / dH < size1 ? pixi / dH : size1 - 1 ;
        int jmax= pixj / dW < size2 ? pixj / dW : size2 - 1 ;

	int i;
	int j;
	int k;

	float gradvalues[maxnumplanes/32];
		for(k=0; k<valuesperthread; k++) {
			gradvalues[k]=0;
		}

	bool zeropad=pixi<padup || pixi>isize1-1+padup || pixj<padleft || pixj>isize2-1+padleft ;
	
	ptrgradinput += gradinputstr0*blockIdx.z + ((pixi-padup) * isize2 + (pixj-padleft)) * nInputPlane ;
	ptrkslices   += kslicesstr0*blockIdx.z + ((imin * size2  + jmin) * kH * kW +  (pixi - imin * dH) * kW + (pixj - jmin*dW) ) * nInputPlane;

	int stridej = (kH*kW - dW) * nInputPlane;
	int stridei = (((size2-jmax+jmin-1)*kH -dH)*kW  + (jmax-jmin+1)*dW)*nInputPlane;

	if(tidx<nInputPlane) {
		if(!zeropad) {
			for(i=imin; i<imax+1; i++) {
				for(j=jmin; j<jmax+1; j++) {
					for(k=0; k<valuesperthread; k++) {
						gradvalues[k] += ptrkslices[k*blk+tidx];
					}
				ptrkslices += stridej;
				}
				ptrkslices += stridei;
			}	
			for(k=0; k<valuesperthread; k++) {
				ptrgradinput[k*blk+tidx] = gradvalues[k];
			}
		}
	}
}
*/



template <int maxnumplanes> __global__ void copyPixelsInSlicesReg(float *ptrinput, float *ptrkslices,
	int dH, int dW, int kH, int kW, int size1, int size2, int isize1, int isize2, int nInputPlane, int valuesperthread, int padleft, int padright, int padup, int paddown, int inputstr0, int kslicesstr0)
{
	const int pixi=blockIdx.x;
	const int pixj=blockIdx.y;
	const int blk =blockDim.x;
	const int tidx=threadIdx.x;

        int imin=(pixi - (kH - 1) + (dH -1))/dH > 0 ? (pixi - (kH - 1) + (dH -1))/dH : 0 ;
        int jmin=(pixj - (kW - 1) + (dW -1))/dW > 0 ? (pixj - (kW - 1) + (dW -1))/dW : 0 ;
        int imax= pixi / dH < size1 ? pixi / dH : size1 - 1 ;
        int jmax= pixj / dW < size2 ? pixj / dW : size2 - 1 ;

	int i;
	int j;
	int k;

	bool zeropad=pixi<padup || pixi>isize1-1+padup || pixj<padleft || pixj>isize2-1+padleft ;
	
	ptrinput   += inputstr0*blockIdx.z + ((pixi-padup) * isize2 + (pixj-padleft)) * nInputPlane ;
	ptrkslices += kslicesstr0*blockIdx.z + ((imin * size2  + jmin) * kH * kW +  (pixi - imin * dH) * kW + (pixj - jmin*dW) ) * nInputPlane;

	float pixvalues[maxnumplanes/32];
	if(tidx<nInputPlane) {
		if (zeropad) 
		{
			for(k=threadIdx.x; k<nInputPlane; k+=blockDim.x) {
				pixvalues[k]=0;
			}
		}
		else
		{
			for(k=threadIdx.x; k<nInputPlane; k+=blockDim.x) {
				pixvalues[k]=ptrinput[k];
			}
		}
	}

	int stridej = (kH*kW - dW) * nInputPlane;
	int stridei = (((size2-jmax+jmin-1)*kH -dH)*kW  + (jmax-jmin+1)*dW)*nInputPlane;
	
		for(i=imin; i<imax+1; i++) {
			for(j=jmin; j<jmax+1; j++) {
				if(zeropad) 
				{
					for(k=threadIdx.x; k<nInputPlane; k+=blockDim.x) {
						ptrkslices[k]=0;
					}
				}
				else {
					for(k=threadIdx.x; k<nInputPlane; k+=blockDim.x) {
						ptrkslices[k]=pixvalues[k];
					}
				}
				ptrkslices += stridej;
			}
			ptrkslices += stridei;
		}	

}


__global__ void copyPixelsInSlicesRGB(float *ptrinput, float *ptrkslices,
	int dH, int dW, int kH, int kW, int size1, int size2, int isize1, int isize2, int nInputPlane, int padleft, int padright, int padup, int paddown, int inputstr0, int kslicesstr0)
{
	// each block does one pixel of the input image
	// each kernel slice is represented by its upper-left coordinates

	const int pixi=blockIdx.x;
	const int pixj=blockIdx.y*blockDim.y + threadIdx.y;
	const int tidx=threadIdx.x;

	int i,j;

	if(pixj > isize2 + padleft + padright -1) return;

	// step 1 : find which kernel slices contain the values of the pixel
        const int imin=(pixi - (kH - 1) + (dH -1))/dH > 0 ? (pixi - (kH - 1) + (dH -1))/dH : 0 ;
        const int jmin=(pixj - (kW - 1) + (dW -1))/dW > 0 ? (pixj - (kW - 1) + (dW -1))/dW : 0 ;
        const int imax= pixi / dH < size1 ? pixi / dH : size1 - 1 ;
        const int jmax= pixj / dW < size2 ? pixj / dW : size2 - 1 ;

	// step 2 : move the pointers
	// this one goes to where the pixel is at
	ptrinput   += inputstr0*blockIdx.z + ((pixi-padup) * isize2 + (pixj-padleft)) * nInputPlane ;
	ptrkslices += kslicesstr0*blockIdx.z + ((imin * size2  + jmin) * kH * kW +  (pixi - imin * dH) * kW + (pixj - jmin*dW) ) * nInputPlane;

	bool zeropad = pixi<padup || pixi>isize1-1+padup || pixj<padleft || pixj>isize2-1+padleft ;
	// read pixel
	// load the stuff first...
	float pixvalue;
	if (zeropad) 	{
		pixvalue=0;
	}
	else	{
		pixvalue=ptrinput[tidx];
	}

	int stridej = (kH*kW - dW) * nInputPlane;
	int stridei = (size2*kH-dH) * kW *nInputPlane - (jmax-jmin+1) * stridej ;

//	write to memory
	for(i=imin; i<imax+1; i++) {
		for(j=jmin; j<jmax+1; j++) {
			if(zeropad) 
			{
				ptrkslices[tidx]=0;
			}
			else {
				ptrkslices[tidx]=pixvalue;
			}
			ptrkslices += stridej;
		}
		ptrkslices += stridei;
	}	
}




__global__ void copyBiasToOutputs(float *ptrbias, float *ptroutput, const int size1, const int size2, const int nOutputPlane, const int linestride, const int imstride)
{
	// each thread has a value to manage...
	//const int blk =blockDim.x;
	const int tidx=blockDim.x*blockIdx.x + threadIdx.x;
	const int tidy=blockIdx.y;
	const int tidz=blockIdx.z;	

	int i;

	float val = ptrbias[tidx];
	ptroutput+= tidz*imstride + tidy*linestride;

	for(int k=0; k<size2; k++)
	{
		if(tidx<nOutputPlane) {
			ptroutput[k*nOutputPlane+tidx]=val;
		}
	}
}


void copyBiasVector(THCudaTensor* output, THCudaTensor* bias)
{
		float* ptrbias    = THCudaTensor_data(bias);
		float* ptroutput  = THCudaTensor_data(output);
		int nOutputPlane	= bias->size[0];
		int batchsize		= output->size[0];
		int size1			= output->size[1];
		int size2			= output->size[2];
  		// fill output with biases
  		dim3 blocksbias ((nOutputPlane+31)/32, size1, batchsize);
  		dim3 threadsbias (32);
  		copyBiasToOutputs<<<blocksbias, threadsbias>>>(ptrbias, ptroutput, size1, size2, nOutputPlane, output->stride[1], output->stride[0]); 
}




__global__ void computeGradBias32(float *ptrgradbias, float *ptrgradoutput, const int size1, const int size2, const int nOutputPlane, float scale, const int batchsize, const int batchstride)
{
	const int tid = blockDim.x*blockIdx.x + threadIdx.x;
	const int tidx = threadIdx.x;
	const int tidy = threadIdx.y;
	const int numpix=size1*size2;
	
	__shared__ float values[32][32];

	float value = 0;
	int i,j;

	for(j=0; j<batchsize; j++)
	{
		for(i=0; i+tidy<numpix; i+=blockDim.y) {
			if (tid<nOutputPlane) {
			value += ptrgradoutput[j*batchstride+(i+tidy)*nOutputPlane+tid];
			}
		}
	}

	values[tidy][tidx]=value;
	__syncthreads();
	// reduction :

	if (tidy == 0) {
		float gradbiasvalue=0;
		#pragma unroll
		for(i=0; i<32;i++){ gradbiasvalue+=values[i][tidx]; }

		if (tid<nOutputPlane) {
			atomicAdd(&ptrgradbias[tid], scale*gradbiasvalue);
		}
	}
	
}




void sliceInput(THCudaTensor *input, THCudaTensor* kernelSlices, int kH, int kW, int dH, int dW, int padup, int paddown, int padleft, int padright)
{
  // find the size of kernelslices
  long batchsize = input->size[0];
  long isize1 = input->size[1];
  long isize2 = input->size[2];
  long nInputPlane = input->size[3];
  long size1 = (isize1 - kH + padup + paddown) / dH + 1;
  long size2 = (isize2 - kW + padleft + padright) / dW + 1;

  float* ptrkslices = THCudaTensor_data(kernelSlices);
  float* ptrinput   = THCudaTensor_data(input);

	int inputstr0=input->stride[0];
	int kslicesstr0=size1*size2*kW*kH*nInputPlane;


  // cuda blocks & threads:
  dim3 blocks (isize1 + padup + paddown, isize2 + padleft + padright, batchsize);
  dim3 threads (32);

	  //with an upper bound on the number of planes, we can be more efficient
	  //kernel unfold inputs
	  if (nInputPlane >1024) {
	  copyPixelsInSlices<<<blocks, threads>>>(ptrinput, ptrkslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, 0, padleft, padright, padup, paddown, inputstr0, kslicesstr0);
	  }
	  else if (nInputPlane >512) {
		copyPixelsInSlicesReg <1024> <<<blocks, threads>>>(ptrinput, ptrkslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, 0, padleft, padright, padup, paddown, inputstr0, kslicesstr0);
	  }
	  else if (nInputPlane >384) {
		copyPixelsInSlicesReg <512> <<<blocks, threads>>>(ptrinput, ptrkslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, 0, padleft, padright, padup, paddown, inputstr0, kslicesstr0);
	  }
	  else if (nInputPlane >256) {
		copyPixelsInSlicesReg <384> <<<blocks, threads>>>(ptrinput, ptrkslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, 0, padleft, padright, padup, paddown, inputstr0, kslicesstr0);
	  }
	  else if (nInputPlane >128) {
		copyPixelsInSlicesReg <256> <<<blocks, threads>>>(ptrinput, ptrkslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, 0, padleft, padright, padup, paddown, inputstr0, kslicesstr0);
	  }
	  else if (nInputPlane >32) {
		copyPixelsInSlicesReg <128> <<<blocks, threads>>>(ptrinput, ptrkslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, 0, padleft, padright, padup, paddown, inputstr0, kslicesstr0);
	  }
	  else if (nInputPlane ==3) {
		  dim3 blocksRGB (isize1 + padup + paddown, (isize2 + padleft + padright+9)/10, batchsize);
		  dim3 threadsRGB (3,10);
		copyPixelsInSlicesRGB <<<blocksRGB, threadsRGB>>>(ptrinput, ptrkslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, padleft, padright, padup, paddown, inputstr0, kslicesstr0);
	  }
	  else {
		copyPixelsInSlicesReg <32> <<<blocks, threads>>>(ptrinput, ptrkslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, 0, padleft, padright, padup, paddown, inputstr0, kslicesstr0);
	  }

  //THCudaTensor_free(input); 


}

void unsliceGradient(THCudaTensor *backwardSlices, THCudaTensor *gradInput, THCudaTensor *gradOutput, int kH, int kW, int dH, int dW, int padup, int paddown, int padleft, int padright)
{

  long batchsize = gradInput->size[0];
  long isize1 = gradInput->size[1];
  long isize2 = gradInput->size[2];
  long nInputPlane = gradInput->size[3];
  long size1 = gradOutput->size[1];
  long size2 = gradOutput->size[2];

  float* ptrbackslices = THCudaTensor_data(backwardSlices);
  float* ptrgradinput  = THCudaTensor_data(gradInput);

  dim3 blocks (isize1 + padup + paddown, isize2 + padleft + padright, batchsize);
  dim3 threads (32);

	int gradinputstr0=gradInput->stride[0];
	int kslicesstr0=size1*size2*kW*kH*nInputPlane;

  // this is for the specific case of the inputs with less than 32 channels
  // for some reason i thought it would be cool to be able to backprop through it

	  if (nInputPlane >1024) {
	  addPixelsInSlices<<<blocks, threads>>>(ptrgradinput, ptrbackslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, padleft, padright, padup, paddown, gradinputstr0, kslicesstr0);
	  }
	  else if (nInputPlane >512)  {
	  addPixelsInSlicesReg <1024> <<<blocks, threads>>>(ptrgradinput, ptrbackslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, padleft, padright, padup, paddown, gradinputstr0, kslicesstr0);
	  } 
	  else if (nInputPlane >384)  {
	  addPixelsInSlicesReg <512> <<<blocks, threads>>>(ptrgradinput, ptrbackslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, padleft, padright, padup, paddown, gradinputstr0, kslicesstr0);
	  } 
	  else if (nInputPlane >256)  {
	  addPixelsInSlicesReg <384> <<<blocks, threads>>>(ptrgradinput, ptrbackslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, padleft, padright, padup, paddown, gradinputstr0, kslicesstr0);
	  } 
	  else if (nInputPlane >128)  {
	  addPixelsInSlicesReg <256> <<<blocks, threads>>>(ptrgradinput, ptrbackslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, padleft, padright, padup, paddown, gradinputstr0, kslicesstr0);
	  } 
	  else if (nInputPlane >32)  {
	  addPixelsInSlicesReg <128> <<<blocks, threads>>>(ptrgradinput, ptrbackslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, padleft, padright, padup, paddown, gradinputstr0, kslicesstr0);
	  } 
	  else {
	  addPixelsInSlicesReg <32> <<<blocks, threads>>>(ptrgradinput, ptrbackslices,
		dH, dW, kH, kW, size1, size2, isize1, isize2, nInputPlane, padleft, padright, padup, paddown, gradinputstr0, kslicesstr0);
	  } 

}




static int cunxn_SpatialConvolutionUnfold_updateOutput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *output = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "output", "torch.CudaTensor");
  THCudaTensor *kernels = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *bias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "bias", "torch.CudaTensor");
//  THCudaTensor *kSlices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "kernelSlices", "torch.CudaTensor");
  long kW = luaT_getfieldcheckint(L, 1, "kW");
  long kH = luaT_getfieldcheckint(L, 1, "kH");
  long dW = luaT_getfieldcheckint(L, 1, "dW");
  long dH = luaT_getfieldcheckint(L, 1, "dH");
  long padup = luaT_getfieldcheckint(L, 1, "padtop");
  long paddown = luaT_getfieldcheckint(L, 1, "padbottom");
  long padleft = luaT_getfieldcheckint(L, 1, "padleft");
  long padright = luaT_getfieldcheckint(L, 1, "padright");
  long nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  long nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  // input should be contiguous already but... well.
  input = THCudaTensor_newContiguous(input);

  // find the size of kernelslices
  long batchsize = input->size[0];
  long isize1 = input->size[1];
  long isize2 = input->size[2];
  long size1 = (isize1 - kH + padup + paddown) / dH + 1;
  long size2 = (isize2 - kW + padleft + padright) / dW + 1;

  THCudaTensor* kernelSlices = THCudaTensor_newWithSize2d(batchsize*size1*size2,kW*kH*nInputPlane);
  sliceInput(input, kernelSlices, kH, kW, dH, dW, padup, paddown, padleft, padright);
  THCudaTensor_resize4d(output, batchsize, size1, size2, nOutputPlane);

	copyBiasVector(output, bias);

  // unfold conv kernels by resizing
  THCudaTensor_resize2d(kernels, nOutputPlane, kW*kH*nInputPlane);
  THCudaTensor_transpose(kernels, NULL, 0, 1);

  // put output in matrix mode
  THCudaTensor_resize2d(output, batchsize* size1* size2, nOutputPlane);

//  printf("sgemm\n");
  THCudaTensor_addmm(output, 1,1, kernelSlices, kernels);

  THCudaTensor_free(kernelSlices); 
  THCudaTensor_transpose(kernels, NULL, 0, 1);


  // check for errors
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in copyPixelsInSlices: %s\n", cudaGetErrorString(err));
    THError("aborting");
  }

  THCudaTensor_resize4d(output, batchsize, size1, size2, nOutputPlane);

  return 1;
}





static int cunxn_SpatialConvolutionUnfold_updateGradInput(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  long kW = luaT_getfieldcheckint(L, 1, "kW");
  long kH = luaT_getfieldcheckint(L, 1, "kH");
  long dW = luaT_getfieldcheckint(L, 1, "dW");
  long dH = luaT_getfieldcheckint(L, 1, "dH");
  long padup = luaT_getfieldcheckint(L, 1, "padtop");
  long paddown = luaT_getfieldcheckint(L, 1, "padbottom");
  long padleft = luaT_getfieldcheckint(L, 1, "padleft");
  long padright = luaT_getfieldcheckint(L, 1, "padright");
  long nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  long nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");

  THCudaTensor *kernels = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradInput = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradInput", "torch.CudaTensor");

  long batchsize = input->size[0];
  long isize1 = input->size[1];
  long isize2 = input->size[2];
  long size1 = gradOutput->size[1];
  long size2 = gradOutput->size[2];

   THCudaTensor_resizeAs(gradInput, input);
	THCudaTensor_fill(gradInput, 0);

  THCudaTensor_resize2d(gradOutput, batchsize*size1* size2, nOutputPlane);

	THCudaTensor* backwardSlices = THCudaTensor_newWithSize2d(batchsize*size1*size2,kW*kH*nInputPlane);

// backprop gradinput into the slices
  THCudaTensor_addmm(backwardSlices, 0, 1, gradOutput, kernels);


// we resize gradOutput back to what it was...
  THCudaTensor_resize4d(gradOutput, batchsize, size1, size2, nOutputPlane);


	unsliceGradient(backwardSlices, gradInput, gradOutput, kH, kW, dH, dW, padup, paddown, padleft, padright);

	THCudaTensor_free(backwardSlices);

  return 1;
}



static int cunxn_SpatialConvolutionUnfold_accGradParameters(lua_State *L)
{
  THCudaTensor *input = (THCudaTensor *)luaT_checkudata(L, 2, "torch.CudaTensor");
  THCudaTensor *gradOutput = (THCudaTensor *)luaT_checkudata(L, 3, "torch.CudaTensor");
  long kW = luaT_getfieldcheckint(L, 1, "kW");
  long kH = luaT_getfieldcheckint(L, 1, "kH");
  long dW = luaT_getfieldcheckint(L, 1, "dW");
  long dH = luaT_getfieldcheckint(L, 1, "dH");
  long padup = luaT_getfieldcheckint(L, 1, "padtop");
  long paddown = luaT_getfieldcheckint(L, 1, "padbottom");
  long padleft = luaT_getfieldcheckint(L, 1, "padleft");
  long padright = luaT_getfieldcheckint(L, 1, "padright");
  long nOutputPlane = luaT_getfieldcheckint(L, 1, "nOutputPlane");
  long nInputPlane = luaT_getfieldcheckint(L, 1, "nInputPlane");
  long zeroGradients = 0; //luaT_getfieldcheckint(L, 1, "zeroGradients");
  float scale = luaL_optnumber(L, 4, 1);

//  THCudaTensor *kernelSlices = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "kernelSlices", "torch.CudaTensor");
  // find the size of kernelslices
  long batchsize = gradOutput->size[0];
  long batchstride = gradOutput->stride[0];
  long isize1 = input->size[1];
  long isize2 = input->size[2];
  long size1 = gradOutput->size[1];
  long size2 = gradOutput->size[2];

  THCudaTensor* kernelSlices = THCudaTensor_newWithSize2d(batchsize*size1*size2,kW*kH*nInputPlane);
  sliceInput(input, kernelSlices, kH, kW, dH, dW, padup, paddown, padleft, padright);

//  THCudaTensor *kernels = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "weight", "torch.CudaTensor");
  THCudaTensor *gradWeight = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradWeight", "torch.CudaTensor");
  THCudaTensor *gradBias = (THCudaTensor *)luaT_getfieldcheckudata(L, 1, "gradBias", "torch.CudaTensor");

//  printf("accgradparameters");

  THCudaTensor_resize2d(gradOutput, batchsize*size1* size2, nOutputPlane);

  float* ptrgradbias = THCudaTensor_data(gradBias);
  float* ptrgradoutput  = THCudaTensor_data(gradOutput);
  dim3 blocksgradbias (nOutputPlane+31/32);
  dim3 threadsgradbias (32,32);

  THCudaTensor_resize2d(gradWeight, nOutputPlane, kW*kH*nInputPlane);
  THCudaTensor_transpose(gradOutput, NULL, 0, 1);

	THCudaTensor_addmm(gradWeight, 1, scale, gradOutput, kernelSlices); 
	computeGradBias32 <<<blocksgradbias, threadsgradbias>>>  (ptrgradbias, ptrgradoutput, size1, size2, nOutputPlane, scale, batchsize, batchstride);

  THCudaTensor_transpose(gradOutput, NULL, 0, 1);
//  THCudaTensor_transpose(gradWeight, NULL, 0, 1);

  THCudaTensor_resize4d(gradWeight, nOutputPlane, kH, kW, nInputPlane);

// we resize gradOutput back to what it was...
  THCudaTensor_resize4d(gradOutput, batchsize, size1, size2, nOutputPlane);
	THCudaTensor_free(kernelSlices);

return 1;

}

static const struct luaL_Reg cunxn_SpatialConvolutionUnfold__ [] = {
  {"SpatialConvolutionUnfold_updateOutput", cunxn_SpatialConvolutionUnfold_updateOutput},
  {"SpatialConvolutionUnfold_updateGradInput", cunxn_SpatialConvolutionUnfold_updateGradInput},
  {"SpatialConvolutionUnfold_accGradParameters", cunxn_SpatialConvolutionUnfold_accGradParameters},
  {NULL, NULL}
};

static void cunxn_SpatialConvolutionUnfold_init(lua_State *L)
{
  luaT_pushmetatable(L, "torch.CudaTensor");
  luaT_registeratname(L, cunxn_SpatialConvolutionUnfold__, "nxn");
  lua_pop(L,1);
}
