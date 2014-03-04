local ConvProto, parent = torch.class('nxn.ConvProto', 'nxn.Module')


function ConvProto:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padleft, padright, padtop, padbottom, overlap)
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
   self.padtop = padtop or 0
   self.padbottom = padbottom or 0
   self.overlap = overlap or 0

   self.alpha= alpha or 1
   self.beta= beta or 0

   self.weight = torch.Tensor(kH, nOutputPlane, kW, nInputPlane)
   self.bias = torch.Tensor(nOutputPlane)

   self.gradWeight = torch.Tensor(kH, nOutputPlane, kW, nInputPlane):zero()
   self.gradBias = torch.Tensor(nOutputPlane):zero()
   
   self:reset()
end

function ConvProto:reset(stdv)
   if stdv then
      stdv = stdv
   else
      stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
   end
   torch.randn(self.weight, self.weight:size())
   self.weight:mul(stdv)
   torch.randn(self.bias, self.bias:size())
   self.bias:mul(stdv)
end

function ConvProto:resetuniform(stdv)
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

function ConvProto:updateOutput(input)
   input.nxn.ConvProto_updateOutput(self, input)
   return self.output
end

function ConvProto:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nxn.ConvProto_updateGradInput(self, input, gradOutput)
      return self.gradInput
   end
end

function ConvProto:zeroGradParameters()
   self.gradWeight:zero()
   self.gradBias:zero()
   -- they will be zeroed during the gradient computation
end

function ConvProto:accGradParameters(input, gradOutput, scale)
    scale = scale or 1
    input.nxn.ConvProto_accGradParameters(self, input, gradOutput, scale) 
--    return 
end



function ConvProto:clipWeights(normbound)
   for idx=1,self.nOutputPlane do
      local filternorm=self.weight:select(2,idx):norm()
      if filternorm > normbound then
         self.weight:select(2,idx):mul(normbound/filternorm)
      end
   end
end

function ConvProto:clipWeights(normbound)
   self.weight.nxn.ConvProto_clipWeights(self, normbound)
end


