
require 'ffi'
require 'torch'
require 'torchffi'

--- direct access to the THBlas GEMM function

local TH=ffi.load("TH")
ffi.cdef([[ void THFloatBlas_gemm(char transa, char transb, long m, long n, long k, 
                      float alpha, float *a, long lda, float *b, long ldb, 
                      float beta, float *c, long ldc);
               void THDoubleBlas_gemm(char transa, char transb, long m, long n, long k, 
                      double alpha, double *a, long lda, double *b, long ldb, 
                      double beta, double *c, long ldc); ]])

function GEMM(type, transa, transb, ...)
   assert(type == "torch.FloatTensor" or type == "torch.DoubleTensor")
   transa = ffi.new('char', string.byte(transa or 'N'))
   transb = ffi.new('char', string.byte(transb or 'N'))
   if type == "torch.FloatTensor" then 
      TH.THFloatBlas_gemm(transa,transb,...)
   else
      TH.THDoubleBlas_gemm(transa,transb,...)
   end 
end



--- utilities

local function newSameTensor(asthis, ...)
   -- create a tensor of same type as <asthis> with the specified dimensions
   return asthis.new(torch.LongStorage{...})
end

local function narrowTensorAndZero(tensor,dim,index,size)
   -- narrow a tensor while clearing the remaining parts
   if index > 1 then
      tensor:narrow(dim,1,index-1):zero()
   end
   if index + size - 1 < tensor:size(dim) then
      tensor:narrow(dim,index + size, tensor:size(dim) - index - size + 1):zero()
   end
   return tensor:narrow(dim,index,size)
end

local function copySpatialConvolutionKernel(kernel, reverse)
   -- copy spatial convolution kernel
   local kh,sh = kernel:size(1), kernel:stride(1)
   local ko,so = kernel:size(2), kernel:stride(2)
   local kw,sw = kernel:size(3), kernel:stride(3)
   local ki,si = kernel:size(4), kernel:stride(4)
   local kp = torch.data(kernel)
   local kcopy = newSameTensor(kernel,kh,ko,kw,ki)
   local kr = torch.data(kcopy)
   local k = 0
   for h=0,kh-1 do
      for o=0,ko-1 do
         for w=0,kw-1 do
            for i=0,ki-1 do
               if reverse then
                  kr[k] = kp[ sh*(kh-1-h) + so*o + sw*(kw-1-w) + si*i ]
               else
                  kr[k] = kp[ sh*h + so*o + sw*w + si*i ]
               end
               k = k + 1
            end
         end
      end
   end
   return kcopy
end


--- spatial convolution

function SpatialConvolution(result, input, kernel, parms)
   
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
      
      -- optional parameters
      parms = parms or {}
   local padleft = parms.padleft or 0; -- input padding
   local padtop = parms.padtop or 0;
   local padright = parms.padright or 0;
   local padbottom = parms.padbottom or 0;
   local stridex = parms.stridex or 1; -- convolution stride
   local stridey = parms.stridey or 1;
   local reverse = parms.reverse or false; -- reversed kernel for backpropagation
   local exact = parms.exact; -- error if padding is not exact
   local overlap = parms.overlap; -- whether blas takes overlapped matrices
   local alpha = parms.alpha or 1; -- result=alpha*result+beta*convolution
   local beta = parms.beta or 0;
   assert(padleft>=0 and padtop>=0 and padright>=0 and padbottom>=0)
   assert(stridex>=1 and stridey>=1)
   
   -- compute output size
   local ow = math.floor((iw + padleft + padright - kw) / stridex) + 1
   local oh = math.floor((ih + padtop + padbottom - kh) / stridey) + 1
   
   -- correct padright and padbottom
   local oldpadright = padright
   local oldpadbottom = padbottom
   padright = ow * stridex + kw - stridex - iw - padleft;
   padbottom = oh * stridey + kh - stridey - ih - padtop;
   assert(not exact or padright ~= oldpadright, "horizontal size mismatch");
   assert(not exact or padbottom ~= oldpadbottom, "horizontal size mismatch");
   if padright < 0 then padright = 0 end
   if padbottom < 0 then padbottom = 0 end
   
   -- input size with padding
   local piw = padleft + iw + padright; 
   local pih = padtop + ih + padbottom;
   
   -- number of horizontal strides between nonoverlapping runs
   local nxs = 1
   if not overlap then
      nxs = math.floor((kw + stridex - 1) / stridex)
   end
   
   -- total size of output buffer
   local tow = math.floor((piw + stridex - 1) / stridex)
   local toh = math.floor((pih + stridey - 1) / stridey)
   
   -- total size of input and output buffers
   local tiw = tow * stridex;
   local tih = toh * stridey;  
   assert(tiw >= piw and piw >= iw)
   assert(tih >= pih and pih >= ih)
   
   -- copy image into input buffer
   local icopy =  newSameTensor(input, stridey, bs, toh, tiw, ip):zero()
   
  
   for s=0,stridey-1 do
      local fout = math.floor((math.max(0,padtop-s)+stridey-1)/stridey)
      local fin = fout * stridey - padtop + s
      assert(fout >= 0 and fin >= 0)
      local icopyptr=torch.data(icopy)
      local inputptr=torch.data(input)
      if fin < ih then
         --inputstride2 = (input:stride())[2]*stridey
         local inputsize2   = math.floor(((ih-fin) + stridey - 1) / stridey)
         local iticopy0=s*bs*toh*tiw*ip
         local itinput0=0
         for it1=1,bs do
            local iticopy1=iticopy0+(it1-1)*toh*tiw*ip
            local itinput1=itinput0+(it1-1)*ih*iw*ip
            
            for it2=1, inputsize2 do
               local iticopy2=iticopy1+(fout+it2-1)*tiw*ip
               local itinput2=itinput1+((fin+1)+(it2-1)*stridey-1)*iw*ip
               
               for it3=1,input:size(3) do
                  local iticopy3=iticopy2+(padleft+it3-1)*ip
                  local itinput3=itinput2+(it3-1)*ip
                  
                  for it4=1,ip do
                     --icopy[{s+1, it1, fout+it2, padleft+it3, it4}]=input[{it1, (fin+1) + (it2-1)*stridey, it3, it4}]
                     --icopyptr[iticopy3]=input[{it1, (fin+1) + (it2-1)*stridey, it3, it4}]
                     icopyptr[iticopy3]=inputptr[itinput3]
                     iticopy3=iticopy3+1
                     itinput3=itinput3+1
                  end
               end
            end
         end
      else
         
         for it=bs*toh*tiw*ip*s, bs*toh*tiw*ip*(s+1)-1 do
            icopyptr[it]=0
         end
      end
   end
   --print('copy time : '..z:time().real)
   
   
   -- let's keep that here so we know what we're doing above...
   if false then   
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
   end
   
   -- copy kernel into kernel buffer
   local kcopy = copySpatialConvolutionKernel(kernel,reverse)
   
   -- allocate and clear output buffer
   local ocopy = newSameTensor(result, bs, toh, tow, op)
   ocopy:fill(0)
   
   -- call GEMM
   for hcall =0,nxs-1 do
      for vcall = 0,kh-1 do
         local sq = math.floor(vcall / stridey)
         local sr = vcall - sq * stridey
--   local icopy =  newSameTensor(input, stridey, bs, toh, tiw, ip)
         local iptr = torch.data(icopy[{sr+1,{},sq+1,hcall*stridex+1,{}}])
         local kptr = torch.data(kcopy:select(1,vcall+1))
         local optr = torch.data(ocopy:select(3,hcall+1))
         local nrun = (bs-1)*toh*tow + oh*tow
         local ngem = math.floor((nrun - hcall) / nxs)
         GEMM(type,'T','N', op, ngem, kw*ip, 
              1, kptr, kw*ip, iptr, nxs*stridex*ip,
              1, optr, nxs*op ) 
      end
   end
   
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
   return result
end

local function copySpatialConvolutionKernelReverse(kernel, stridex, stridey)
   local kh,sh = kernel:size(1), kernel:stride(1)
   local ko,so = kernel:size(2), kernel:stride(2)
   local kw,sw = kernel:size(3), kernel:stride(3)
   local ki,si = kernel:size(4), kernel:stride(4)
   local kp = torch.data(kernel)
   
   local kouth=math.floor((kh+stridey-1)/stridey)
   local kouto=ki
   local koutw=math.floor((kw+stridex-1)/stridex)
   local kouti=ko
   
   local kout = newSameTensor(kernel, stridey, stridex, kouth, kouto, koutw, kouti)
   kout:zero()
   local koptr = torch.data(kout)
   
   local i=0
   
   for stry=1,stridey do
      for strx=1,stridex do
         for ith=1, kouth do
            local ycoord=kh+1-((ith-1)*stridey+1+stry-1)
            if ycoord<kh+1 and ycoord>0 then
               for ito=1, kouto do
                  for itw=1, koutw do
                     local xcoord=kw+1-((itw-1)*stridex+1+strx-1)
                     if xcoord<kw+1 and xcoord>0 then
                        for iti=1, kouti do
                           --kout[{stry, strx, ith, ito, itw, iti}] = kernel[{ycoord, iti, xcoord, ito}]
                           koptr[i] = kp[(ycoord-1)*sh+(iti-1)*so+(xcoord-1)*sw+(ito-1)*si]
                           i=i+1
                        end
                     else
                        i = i + kouti
                     end
                  end
               end
            else
               i = i + kouti*koutw*kouto
            end
         end
      end
   end
   
   
   return kout
   
   
end




function ReverseConvolution3(gradInput, gradOutput, input, kernel, parms)
   
   -- typecheck
   local type = torch.typename(input)
   assert(type == "torch.FloatTensor" or type == "torch.DoubleTensor")
   assert(torch.typename(kernel) == type)
   assert(torch.typename(gradInput) == type)
   assert(torch.typename(gradOutput) == type)
   
   -- input sizes
   assert(input:nDimension() == 4)
   bs = input:size(1) -- batch size
   ih = input:size(2) -- input height
   iw = input:size(3) -- input width
   ip = input:size(4) -- input planes
      
   -- kernel
   assert(kernel:nDimension() == 4)
   kh = kernel:size(1) -- kernel height
   op = kernel:size(2) -- output planes
   kw = kernel:size(3) -- kernel width
   assert(ip == kernel:size(4)) -- input planes
            
   -- optional parameters
   parms = parms or {}
   padleft = parms.padleft or 0; -- input padding
   padtop = parms.padtop or 0;
   padright = parms.padright or 0;
   padbottom = parms.padbottom or 0;
   stridex = parms.stridex or 1; -- convolution stride
   stridey = parms.stridey or 1;
   reverse = parms.reverse or false; -- reversed kernel for backpropagation
   exact = parms.exact; -- error if padding is not exact
   overlap = parms.overlap; -- whether blas takes overlapped matrices
   alpha = parms.alpha or 1; -- result=alpha*result+beta*convolution
   beta = parms.beta or 0;
   assert(padleft>=0 and padtop>=0 and padright>=0 and padbottom>=0)
   assert(stridex>=1 and stridey>=1)   

   -- gradOutput sizes
   assert(gradOutput:nDimension() == 4)
   assert(bs == gradOutput:size(1)) -- batch size
   -- check that output h,w sizes match gradOutput sizes      
   goh = gradOutput:size(2)
   gow = gradOutput:size(3)
   assert(goh == math.floor((ih + padtop + padbottom - kh) / stridey) + 1) 
      assert(gow == math.floor((iw + padleft + padright - kw) / stridex) + 1) 
      assert(op == gradOutput:size(4))
   
   
   
   revk = copySpatialConvolutionKernelReverse(kernel, stridex, stridey)
   revkh = revk:size(3)
   revkw = revk:size(5)
   
   
   
   -- create gradinput tensor :
   giw = ( gow + revkw -1 ) * stridex
   gih = ( goh + revkh -1 ) 
   gradin = newSameTensor(gradOutput, stridey, bs, gih, giw, ip)
   gradin:zero()
   
   
   
   -- pad gradoutput tensor :
   pgow = ( gow + revkw -1 )
   pgoh = ( goh + revkh -1 ) 

   
      -- here we take bs+1 to have some zero-padding at the end of the matrix
      -- it only costs some memory. GEMM does not use it.
   gradOutCopy = newSameTensor(gradOutput, bs+1, pgoh, pgow, op) 
   tgocopy=narrowTensorAndZero(gradOutCopy, 1, 1, bs)
   tgocopy=narrowTensorAndZero(tgocopy, 2, revkh, goh)
   tgocopy=narrowTensorAndZero(tgocopy, 3, revkw, gow)
   tgocopy:copy(gradOutput)
   
   
  --GEMM call :
   nxs=1
   if not overlap then
      nxs=revkw
      print('no overlap')
   end
   for hcall =0,nxs-1 do
      for stry=1,stridey do
         for strx=1,stridex do
            for vcall=1,revkh do
               --            for hcall=1,revkw do
               --gradoutptr = torch.data(gradOutCopy[{1, revkh-(vcall-1), 1, {}}])
               gradoutptr = torch.data(gradOutCopy[{1, revkh-(vcall-1), hcall+1, {}}])
               ldgradout  = op *nxs
               --ldgradout  = op --overlapping
               
               krevptr    = torch.data(revk[{stry,strx,revkh-(vcall-1),{},{},{}}])
               szkrev     = op*revkw
               ldkrev     = op*revkw --*revkh 
                  
                  gradinptr  = torch.data(gradin[{stry, 1, 1, stridex-strx+1+hcall*stridex, {}}])
               --gradinptr  = torch.data(gradin[{stry, 1, 1, stridex-strx+1, {}}]) --overlapping
               
               ldgradin   = ip *stridex*nxs
               --ldgradin   = ip *stridex --overlapping
               
               nspots     = giw/stridex*gih*bs
               ngem       = math.ceil((nspots - hcall) / nxs) --overlapping
               --print('overlap mode : off2')
               
               -- GEMM(type, 'T', 'N', ip, nspots, szkrev, 1, krevptr, ldkrev, gradoutptr, ldgradout, 1, gradinptr, ldgradin)  --overlapping           
               GEMM(type, 'T', 'N', ip, ngem, szkrev, 1, krevptr, ldkrev, gradoutptr, ldgradout, 1, gradinptr, ldgradin)           
                  --            end    
            end
         end
      end
   end
   
   if false then
      if not overlap then
         nxs = math.floor((kw + stridex - 1) / stridex)
      end
      
      -- call GEMM
      for hcall =0,nxs-1 do
         for vcall = 0,kh-1 do
            local sq = math.floor(vcall / stridey)
            local sr = vcall - sq * stridey
            --   local icopy =  newSameTensor(input, stridey, bs, toh, tiw, ip)
            local iptr = torch.data(icopy[{sr+1,{},sq+1,hcall*stridex+1,{}}])
            local kptr = torch.data(kcopy:select(1,vcall+1))
            local optr = torch.data(ocopy:select(3,hcall+1))
            local nrun = (bs-1)*toh*tow + oh*tow
            local ngem = math.floor((nrun - hcall) / nxs)
            GEMM(type,'T','N', op, ngem, kw*ip, 
                 1, kptr, kw*ip, iptr, nxs*stridex*ip,
                 1, optr, nxs*op ) 
         end
      end
   end
   
   -- correct padright and padbottom
   local oldpadright = padright
   local oldpadbottom = padbottom
   padright = gow * stridex + kw - stridex - iw - padleft;
   padbottom = goh * stridey + kh - stridey - ih - padtop;
   assert(not exact or padright ~= oldpadright, "horizontal size mismatch");
   assert(not exact or padbottom ~= oldpadbottom, "horizontal size mismatch");
   if padright < 0 then padright = 0 end
   if padbottom < 0 then padbottom = 0 end
   
   -- input size with padding
   local piw = padleft + iw + padright; 
   local pih = padtop + ih + padbottom;
   
   
   
   

   
   throwawayx=stridex - math.mod(kw,stridex)
   throwawayy=stridey - math.mod(kh,stridey)
   if stridex==1 then throwawayx=0 end
   if stridey==1 then throwawayy=0 end
   if throwawayx==stridex then throwawayx=0 end
   if throwawayy==stridey then throwawayy=0 end
   

   resw=piw
   resh=pih
   result = newSameTensor(gradOutput, bs, resh, resw, ip):fill(0)
   

   for stry=stridey,1,-1 do   
      -- copy is tricky
      -- first line should be thrown away if 
      throwaway = stridey-stry < throwawayy
      
      tgicopy = gradin:select(1,stry)
      tgicopy=tgicopy:narrow(3, 1+throwawayx, giw-throwawayx)
      if throwaway then
         tgicopy=tgicopy:narrow(2, 2, gih-1)
      end
      
      
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
   
end







function ComputeGradWeights(gradOutput, input, kernel, parms)
   gradweight = kernel:clone():zero()
   
   
   
end


















-------------------------------------
-- test1: simple kernel

function test1(iw,ih,kw,kh,parm)
   a = torch.rand(ih,iw)
   k = torch.rand(kh,kw)
   c = torch.xcorr2(a,k,'V')
   al = a:reshape(1,ih,iw,1)
   kl = k:reshape(kh,1,kw,1)
   cl = c:clone()
   SpatialConvolution(cl,al,kl,parm)
   print(c)
   print(cl[{1,{},{},1}])
end

-- test1(4,7,3,2)

-------------------------------------
-- test2: reversed kernel

function test2(iw,ih,kw,kh)
   a = torch.rand(ih,iw)
   k = torch.rand(kh,kw)
   c = torch.conv2(a,k,'V')
   al = a:reshape(1,ih,iw,1)
   kl = k:reshape(kh,1,kw,1)
   cl = c:clone()
   SpatialConvolution(cl,al,kl,{reverse=true})
   print(c)
   print(cl[{1,{},{},1}])
end

-- test2(4,7,3,2)

-------------------------------------
-- test3: stridex

function test3(iw,ih,kw,kh)
   a = torch.rand(ih,iw)
   k = torch.rand(kh,kw)
   c = torch.xcorr2(a,k,'V') -- does not have stride. take every other column
      al = a:reshape(1,ih,iw,1)
   kl = k:reshape(kh,1,kw,1)
   cl = c:clone()
   SpatialConvolution(cl,al,kl,{stridex=2})
   print(c)
   print(cl[{1,{},{},1}])
end

-- test3(8,4,3,2)

-------------------------------------
-- test4: stridey

function test4(iw,ih,kw,kh)
   a = torch.rand(ih,iw)
   k = torch.rand(kh,kw)
   c = torch.xcorr2(a,k,'V') -- does not have stride. take every other row
      al = a:reshape(1,ih,iw,1)
   kl = k:reshape(kh,1,kw,1)
   cl = c:clone()
   SpatialConvolution(cl,al,kl,{stridey=2})
   print(c)
   print(cl[{1,{},{},1}])
end

-- test4(4,9,3,7)


-------------------------------------
-- to be tested
-- * padding
-- * multiple input/output planes 
-- * mini-batches
-- * bound checks with valgrind


-------------------------------------
-- challenge
-- * C version with zero based indices
