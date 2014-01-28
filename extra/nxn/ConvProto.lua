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

   self.gradWeight = torch.Tensor(kH, nOutputPlane, kW, nInputPlane)
   self.gradBias = torch.Tensor(nOutputPlane)
   
   self:reset()
end

function ConvProto:reset(stdv)
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
   self.zeroGradients = 1
   -- they will be zeroed during the gradient computation
end

function ConvProto:accGradParameters(input, gradOutput, scale)
    input.nxn.ConvProto_accGradParameters(self, input, gradOutput, scale) 
    self.zeroGradients = 0
--    return 
end
