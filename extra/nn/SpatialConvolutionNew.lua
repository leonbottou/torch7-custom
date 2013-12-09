local SpatialConvolutionNew, parent = torch.class('nn.SpatialConvolutionNew', 'nn.Module')

function SpatialConvolutionNew:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, shdmem, padleft, padright, padup, paddown)
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
   self.shdmem = shdmem or 1
   self.kernelSlices = torch.Tensor()
   self.backwardSlices = torch.Tensor()


   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)

-- zeroGradParameters will turn this to 1 and the next gradient 
-- computation will flush the accumulated gradients
   self.zeroGradients = 0 
   self.gradWeight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.gradBias = torch.Tensor(nOutputPlane)
   
   self:reset()
end

function SpatialConvolutionNew:reset(stdv)
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

function SpatialConvolutionNew:updateOutput(input)
   input.nn.SpatialConvolutionNew_updateOutput(self, input)
   return self.output
end

function SpatialConvolutionNew:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nn.SpatialConvolutionNew_updateGradInput(self, input, gradOutput)
      return self.gradInput
   end
end

function SpatialConvolutionNew:zeroGradParameters()
   self.zeroGradients = 1
   -- they will be zeroed during the gradient computation
end

function SpatialConvolutionNew:accGradParameters(input, gradOutput, scale)
    input.nn.SpatialConvolutionNew_accGradParameters(self, input, gradOutput, scale) 
    self.zeroGradients = 0
--    return 
end
