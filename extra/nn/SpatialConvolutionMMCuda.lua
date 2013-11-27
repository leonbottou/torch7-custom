local SpatialConvolutionMMCuda, parent = torch.class('nn.SpatialConvolutionMMCuda', 'nn.Module')


local function zaptensor(x)
   x:resize(torch.LongStorage())
   x:storage():resize(0)
end
   

-- this module only works in forward mode, for now

function SpatialConvolutionMMCuda:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   
   self:reset()
   

end

function SpatialConvolutionMMCuda:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nn.oldSeed then
      self.weight:apply(function()
         return torch.uniform(-stdv, stdv)
      end)
      self.bias:apply(function()
         return torch.uniform(-stdv, stdv)
      end) 
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function SpatialConvolutionMMCuda:updateOutput(input)
   -- we have to do that or the resize will mess up the weights (not in cuda mode though)
   -- it will only happen on first run though
   self.weight=self.weight:contiguous()
   -- get kernels ready
   -- these are the sizes of self.weight
   local weightsizes=torch.LongStorage({self.nOutputPlane, self.nInputPlane, self.kH, self.kW})

   local matkernels=self.weight
   matkernels:resize(weightsizes[1],weightsizes[2]*weightsizes[3]*weightsizes[4])
   matkernels=matkernels:transpose(1,2)
   
  
   local matinput=input:unfold(2,(weightsizes)[3],self.dH):unfold(3,(weightsizes)[4],self.dW):transpose(1,2):transpose(2,3)

   local Wsize=(#matinput)[1]
   local Hsize=(#matinput)[2]
   
   -- maxwslice is here to avoid bad invalid argument (too many blocks/threads in the newmat:contiguous() call...)
   local maxwslice=math.floor(4651200/((#matinput)[2]*(#matinput)[3]*(#matinput)[4]*(#matinput)[5]))
   
  -- self.output=input.new(nOutputPlane, Wsize, Hsize)
   self.output:resize(self.nOutputPlane, Wsize, Hsize)
   for i = 1,Wsize,maxwslice do
      local sliceWstart=i
      local sliceWend  =math.min(i+maxwslice, Wsize)
      local sliceW     =sliceWend-sliceWstart+1
      
      local newmat=matinput[{{sliceWstart, sliceWend},{},{},{},{}}]
      local newmatcontiguous = newmat:isContiguous()
      newmat=newmat:contiguous()
      newmat:resize(Hsize*sliceW,(#matinput)[3]*(#matinput)[4]*(#matinput)[5])
      
      local res=input.new((#matkernels)[2],(#newmat)[1]):zero()
      
      -- addr doesn't like it when the tensor is a vector...
      if (#res)[2]==1 then
         res:copy(self.bias)
         res:resize((#matkernels)[2],(#newmat)[1])
      else
         local tmpmat=input.new((#res)[2]):fill(1)
         res:addr(self.bias,tmpmat)
      end
      
      res=res:transpose(1,2)      
      res=res:addmm(newmat,matkernels):transpose(1,2)
      res:resize((#matkernels)[2],sliceW,Hsize)
      
      local outselect=self.output:narrow(2,sliceWstart, sliceW)
      outselect:copy(res)
      
      if not newmatcontiguous then zaptensor(newmat) end
      if tmpmat then zaptensor(tmpmat) end
      zaptensor(res)
      
   end
   
   self.weight:resize(weightsizes)
   cutorch.synchronize()
   return self.output
end

function SpatialConvolutionMMCuda:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialConvolutionMMCuda_updateGradInput(self, input, gradOutput)
   end
end

function SpatialConvolutionMMCuda:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolutionMMCuda_accGradParameters(self, input, gradOutput, scale)
end
