local SpatialConvolutionUnfold, parent = torch.class('nxn.SpatialConvolutionUnfold', 'nxn.Module')

function SpatialConvolutionUnfold:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padleft, padright, padup, paddown)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.padleft = padleft or 0
   self.padright = padright or 0
   self.padup = padup or 0
   self.paddown = paddown or 0
   self.kernelSlices = torch.Tensor()
   self.backwardSlices = torch.Tensor()


   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)

-- zeroGradParameters will turn this to 1 and the next gradient 
-- computation will flush the accumulated gradients
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW):zero()
   self.gradBias = torch.Tensor(nOutputPlane):zero()
   
   self:reset()
end

function SpatialConvolutionUnfold:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   if nxn.oldSeed then
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

function SpatialConvolutionUnfold:updateOutput(input)
   input.nxn.SpatialConvolutionUnfold_updateOutput(self, input)
   return self.output
end

function SpatialConvolutionUnfold:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nxn.SpatialConvolutionUnfold_updateGradInput(self, input, gradOutput)
      return self.gradInput
   end
end

function SpatialConvolutionUnfold:zeroGradParameters()
	self.gradWeight:zero()
	self.gradBias:zero()
   -- they will be zeroed during the gradient computation
end

function SpatialConvolutionUnfold:accGradParameters(input, gradOutput, scale)
    input.nxn.SpatialConvolutionUnfold_accGradParameters(self, input, gradOutput, scale) 
end
