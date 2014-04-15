local ConvLayer, parent = torch.class('nxn.ConvLayer', 'nxn.Module')


function ConvLayer:__init(nInputPlane, nOutputPlane, kW, kH, dW, dH, padleft, padright, padtop, padbottom)
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
   self.addgrads=0

   self.alpha= alpha or 1
   self.beta= beta or 0

   self.weight = torch.Tensor(kH, nOutputPlane, kW, nInputPlane)
   self.bias = torch.Tensor(nOutputPlane)

   self.gradWeight = torch.Tensor(kH, nOutputPlane, kW, nInputPlane):zero()
   self.gradBias = torch.Tensor(nOutputPlane):zero()
   
   self:reset()
   
   self.mode='conv' -- can be : 'conv', 'trivial', 'fc'
   self.propagate=true
   self.learningrate=0
end

function ConvLayer:setLearningRate(lr)
   if lr > 0 or lr==0 then
      self.learningrate=lr
   else
      error('learning rate must be positive or 0')   
   end
end

function ConvLayer:reset(stdv)
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








-- these functions are necessary because switching from one mode to another
-- requires a hard transpose.
-- should stick to a standard format, and copy kernels during convolution forward step
-- but well...

function ConvLayer:switchToFC()
   if self.mode=='fc' then return end
   if self.mode=='conv' then 
      self.weight=self.weight:transpose(1,2):contiguous()
      self.gradWeight=self.gradWeight:transpose(1,2):contiguous()
      assert(self.weight:size(1)==self.nOutputPlane) -- just checkin'
      self.mode='fc'
--      print('switched layer to dense mode (kernel size == image size, no padding)')
      return
   end
end

function ConvLayer:switchToConv()
   if self.mode=='conv' then return end
   if self.mode=='fc' then 
      self.weight=self.weight:transpose(1,2):contiguous()
      self.gradWeight=self.gradWeight:transpose(1,2):contiguous()
      assert(self.weight:size(1)==self.kH) -- just checkin'
      self.mode='conv'
--      print('switched layer to convolution mode (kernel size < image size)')
      return
   end
end

function ConvLayer:switchToTrivial()
   if self.mode=='trivial' then return end
   assert(self.kH==1 and self.kW==1 and self.dH==1 and self.dW==1)
   self.weight=self.weight:contiguous()
   self.gradWeight=self.gradWeight:contiguous()
   self.weight=self.weight:resize(self.nOutputPlane, self.nInputPlane)
   self.gradWeight=self.gradWeight:resize(self.nOutputPlane, self.nInputPlane)
   self.mode='trivial' -- no-return point.
--   print('switched layer to trivial mode (1x1 kernel, no padding)')
end

function ConvLayer:optimize(input)
   if self.padleft==0 and 
      self.padright==0 and 
      self.padtop==0 and 
      self.padbottom==0 and
      self.kH==1 and 
      self.kW==1 and 
      self.dH==1 and 
      self.dW==1 then 
      -- critical step : add pcall here
      self:switchToTrivial()
      return
   end
   if self.padleft==0 and 
      self.padright==0 and 
      self.padtop==0 and 
      self.padbottom==0 and 
      input:size(2)==self.kH and
      input:size(3)==self.kW and
      input:size(4)==self.nInputPlane then 
      -- critical step : add pcall here
      self:switchToFC()
      return
   end 
      -- critical step : add pcall here
      self:switchToConv()
   return
end







-- update outputs

function ConvLayer:updateOutputTrivial(input)
   -- input is flattened (view)
   local tinput=input.new()
   tinput:set(input:storage(), 1, torch.LongStorage{input:size(1)*input:size(2)*input:size(3), input:stride(3)})
   
   -- weight is flattened (view)
   local tweight=self.weight
   
   -- MM
   self.output:resize(input:size(1)*input:size(2)*input:size(3), self.weight:size(1))
   self.output:zero():addr(1, input.new(input:size(1)*input:size(2)*input:size(3)):fill(1), self.bias)
   self.output:addmm(1, tinput, tweight:t())

   -- output is unflattened
   self.output:resize(input:size(1), input:size(2), input:size(3), self.weight:size(1))
--   print('updateOutputTrivial')
end

function ConvLayer:updateOutputFC(input)
   -- input is flattened (view)
   local tinput=input.new()
   tinput:set(input:storage(), 1, torch.LongStorage{input:size(1), input:stride(1)})
   
   -- weight is flattened (view)
   local tweight=self.weight.new()
   tweight:set(self.weight:storage(), 1, torch.LongStorage{self.weight:size(1), self.weight:stride(1)})
   
   -- MM
   self.output:resize(input:size(1), self.weight:size(1))
   self.output:zero():addr(1, input.new(input:size(1)):fill(1), self.bias)
   self.output:addmm(1, tinput, tweight:t())
   self.output:resize(input:size(1), 1, 1, self.weight:size(1))
--   print('updateOutputFC')
end

function ConvLayer:updateOutputConv(input)
   input.nxn.SpatialConvolution_updateOutput(self, input)
--   print('updateOutputConv')
end

function ConvLayer:updateOutput(input)
   self:optimize(input)
   if self.mode=='trivial' then self:updateOutputTrivial(input) end
   if self.mode=='fc' then self:updateOutputFC(input) end
   if self.mode=='conv' then self:updateOutputConv(input) end
   return self.output
end







-- update gradients

function ConvLayer:updateGradInputTrivial(input, gradOutput)
   -- gradOutput is flattened (view)
   local tgradOutput=gradOutput.new()
   tgradOutput:set(gradOutput:storage(), 1, torch.LongStorage{gradOutput:size(1)*gradOutput:size(2)*gradOutput:size(3), gradOutput:stride(3)})
  
   local tweight=self.weight

   local nElement = self.gradInput:nElement()
   self.gradInput:resizeAs(input)

   self.gradInput:resizeAs(input)
   if self.gradInput:nElement() ~= nElement then
      self.gradInput:zero()
   end

   -- gradInput is flattened (view)
   local tgradInput=self.gradInput.new()
   tgradInput:set(self.gradInput:storage(), 1, torch.LongStorage{self.gradInput:size(1)*self.gradInput:size(2)*self.gradInput:size(3), self.gradInput:stride(3)})

   tgradInput:addmm(0, 1, tgradOutput, tweight)
   
end

function ConvLayer:updateGradInputFC(input, gradOutput)
   -- gradOutput is flattened (view)
   local tgradOutput=gradOutput.new()
   tgradOutput:set(gradOutput:storage(), 1, torch.LongStorage{gradOutput:size(1), gradOutput:stride(1)})
   
   -- weight is flattened (view)
   local tweight=self.weight.new()
   tweight:set(self.weight:storage(), 1, torch.LongStorage{self.weight:size(1), self.weight:stride(1)})
   
   local nElement = self.gradInput:nElement()
   self.gradInput:resizeAs(input)
   if self.gradInput:nElement() ~= nElement then
      self.gradInput:zero()
   end

   -- gradInput is flattened (view)
   local tgradInput=self.gradInput.new()
   tgradInput:set(self.gradInput:storage(), 1, torch.LongStorage{self.gradInput:size(1), self.gradInput:stride(1)})

   tgradInput:addmm(0, 1, tgradOutput, tweight)

end

function ConvLayer:updateGradInputConv(input, gradOutput)
   input.nxn.SpatialConvolution_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function ConvLayer:updateGradInput(input, gradOutput)
   self:optimize(input)
   if self.propagate then 
      if self.mode=='trivial' then self:updateGradInputTrivial(input, gradOutput) end
      if self.mode=='fc' then self:updateGradInputFC(input, gradOutput) end
      if self.mode=='conv' then self:updateGradInputConv(input, gradOutput) end
   end
end







-- update weight gradients

function ConvLayer:zeroGradParameters()
   self.gradWeight:zero()
   self.gradBias:zero()
end

function ConvLayer:accGradParametersTrivial(input, gradOutput, scale)
   -- input is flattened (view)
   local tinput=input.new()
   tinput:set(input:storage(), 1, torch.LongStorage{input:size(1)*input:size(2)*input:size(3), input:stride(3)})
   -- gradOutput is flattened (view)
   local tgradOutput=gradOutput.new()
   tgradOutput:set(gradOutput:storage(), 1, torch.LongStorage{gradOutput:size(1)*gradOutput:size(2)*gradOutput:size(3), gradOutput:stride(3)})
   
   self.gradWeight:addmm(scale, tgradOutput:t(), tinput)
   self.gradBias:addmv(scale, tgradOutput:t(), tinput.new(input:nElement()/self.weight:size(2)):fill(1))
      
end

function ConvLayer:accGradParametersFC(input, gradOutput, scale)
   -- input is flattened (view)
   local tinput=input.new()
   tinput:set(input:storage(), 1, torch.LongStorage{input:size(1), input:stride(1)})

   -- gradOutput is flattened (view)
   local tgradOutput=gradOutput.new()
   tgradOutput:set(gradOutput:storage(), 1, torch.LongStorage{gradOutput:size(1), gradOutput:stride(1)})
   
   -- weight is flattened (view)
   local tgradWeight=self.gradWeight.new()
   tgradWeight:set(self.gradWeight:storage(), 1, torch.LongStorage{self.gradWeight:size(1), self.gradWeight:stride(1)})

   tgradWeight:addmm(scale, tgradOutput:t(), tinput)
   self.gradBias:addmv(scale, tgradOutput:t(), tinput.new(input:size(1)):fill(1))

end

function ConvLayer:accGradParametersConv(input, gradOutput, scale)
   input.nxn.SpatialConvolution_accGradParameters(self, input, gradOutput, scale) 
end

function ConvLayer:accGradParameters(input, gradOutput, scale)
   self:optimize(input)
   scale = scale or 1
   if self.learningrate > 0 then 
      if self.mode=='trivial' then self:accGradParametersTrivial(input, gradOutput, scale) end
      if self.mode=='fc' then self:accGradParametersFC(input, gradOutput, scale) end
      if self.mode=='conv' then self:accGradParametersConv(input, gradOutput, scale) end
   end
--    return 
end



-- clip the weights (this is for later)

function ConvLayer:clipWeights(normbound)
   for idx=1,self.nOutputPlane do
      local filternorm=self.weight:select(2,idx):norm()
      if filternorm > normbound then
         self.weight:select(2,idx):mul(normbound/filternorm)
      end
   end
end

function ConvLayer:clipWeights(normbound)
   self.weight.nxn.ConvLayer_clipWeights(self, normbound)
end


