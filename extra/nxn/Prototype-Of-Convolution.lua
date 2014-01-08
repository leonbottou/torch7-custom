
require 'ffi'
require 'torch'
require 'torchffi'

--- direct access to the THBlas GEMM function

local TH=ffi.load("TH")
ffi.cdef([[ void THFloatBlas_gemm(char transa, char transb, long m, long n, long k, 
                      float alpha, float *a, long lda, 
                      float *b, long ldb, double beta, 
                      float *c, long ldc);
               void THDoubleBlas_gemm(char transa, char transb, long m, long n, long k, 
                       double alpha, double *a, long lda, 
                       double *b, long ldb, double beta, 
                       double *c, long ldc); ]])

function GEMM(type, ...)
   assert(type == "torch.FloatTensor" or type == "torch.DoubleTensor")
   if type == "torch.FloatTensor" then 
      TH.THFloatBlas_gemm(...)
   else
      TH.THRealBlas_gemm(...)
   end 
end



--- spatial convolution

function SpatialConvolutionAcc(result, input, kernel, parms)
   
   -- typecheck
   local type = torch.typename(input)
   assert(type == "torch.FloatTensor" or type == "torch.DoubleTensor")
   assert(torch.typename(kernel) == type)
   assert(torch.typename(result) == type)
   
   -- input
   assert(input:nDimension() == 4)
   local bs = input:size(1) -- batch size
   local ih = input:size(2) -- input height
   local iw = input:size(3) -- input width
   local ip = input:size(4) -- input planes
      
   -- kernel
   assert(kernel:nDimension() == 4)
   local kh = kernel:size(1) -- kernel height
   local op = kernel:size(2) -- output planes
   local kw = kernel:size(3) -- kernel width
   assert(ip == kernel:size(4)) -- input planes
      
   -- parameters
   parms = parms or {}
   local padleft = parms.padleft or 0; -- input padding
   local padtop = parms.padtop or 0;
   local padright = parms.padright or 0;
   local padbottom = parms.padbottom or 0;
   local stridex = parms.stridex or 1; -- convolution stride
   local stridey = parms.stridey or 1;
   local reverse = parms.reverse or false; -- reversed kernel for backpropagation
   local exact = parms.exact or false; -- error if padding is not exact
   
   -- compute output size
   local ow = math.floor((iw + padleft + padright - kw) / stridex) + 1
   local oh = math.floor((ih + padtop + padbottom - kh) / stridex) + 1
   
   -- correct padright and padbottom
   local oldpadright = padright
   local oldpadbottom = padbottom
   padright = padleft + iw - ow * stridex + kw - ow;
   padbottom = padtop + ih - oh * stridey + kh - oh;
   assert(not exact or padright ~= oldpadright, "horizontal size mismatch") 
   assert(not exact or padbottom ~= oldpadbottom, "horizontal size mismatch") 
   local piw = padleft + iw + padright;
   local pih = padtop + ih + padbottom;
   
   -- number of horizontal strides between nonoverlapping runs
   local nxs = math.floor((kw + stridex - 1) / stridex)
   
   -- number of nonoverlapping runs to clear the padded width
   local nxn = math.floor((piw + nxs * stridex - 1) / (nxs * stridex))
   
   -- total padded row size
   local tow = nxn * nxs
   local tiw = tow * stridex
   
   -- number of vertical strides to clear the padded image height
   local nyn = math.floor((pih + stridey - 1) / stridey)
   
   -- total padded vertical image size
   local toh = nyn   
      
   -- copy input into contiguous padded memory chunk
   local icopy =  input:new(stridey * bs * toh * tiw * ip)
   local iptr = torch.data(iptr)
   do 
      ---- TODO
   end
   
   -- copy kernel into contiguous memory chunk
   local kcopy = kernel:new(kernel:nElement())
   local kptr = torch.data(kcopy)
   do
      local kh,sh = kernel.size(1), kernel.stride(1)
      local ko,so = kernel.size(2), kernel.stride(2)
      local kw,sw = kernel.size(3), kernel.stride(3)
      local ki,si = kernel.size(4), kernel.stride(4)
      local kp = torch.data(kernel)
      local k = 0
      for h=1,kh do
         for o=1,ko do
            for w=1,kw do
               for i=1,ki do
                  if reverse then
                     kptr[k] = kp[ sh*(kh-h+1) + so*o + sw*(kw-w+1) + si*i ]
                  else
                     kptr[k] = kp[ sh*h + so*o + sw*w + si*i ]
                  end
                  k = k + 1
               end
            end
         end
      end
   end

   -- allocate and clear contiguous padded memory for output
   local ocopy = output:new(bs * toh * tow * op)
   local optr = torch.data(ocopy)
   ocopy:zero()
      
   -- call GEMM
   ---- TODO 
      
   -- accumulate output chunk into result tensor
   do
      result:resize(bs,oh,ow,op)
      local rptr = torch.data(result)
      ---- TODO 
   end
   return result
end