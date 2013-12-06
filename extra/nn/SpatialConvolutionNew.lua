local SpatialConvolutionNew, parent = torch.class('nn.SpatialConvolutionNew', 'nn.Module')

function SpatialConvolutionNew:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, shdmem)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.nInputPlane = nInputPlane
   self.nOutputPlane = nOutputPlane
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH
   self.shdmem = shdmem or 1
   self.kslicestest = torch.Tensor()

   self.weight = torch.Tensor(nOutputPlane, nInputPlane, kH, kW)
   self.bias = torch.Tensor(nOutputPlane)
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
   return input.nn.SpatialConvolutionNew_updateOutput(self, input)
end

function SpatialConvolutionNew:updateGradInput(input, gradOutput)
   if self.gradInput then
      return input.nn.SpatialConvolutionNew_updateGradInput(self, input, gradOutput)
   end
end

function SpatialConvolutionNew:accGradParameters(input, gradOutput, scale)
   return input.nn.SpatialConvolutionNew_accGradParameters(self, input, gradOutput, scale)
end
