local Module = torch.class('nxn.Module')

function Module:__init()
   self.gradInput = torch.Tensor()
   self.output = torch.Tensor()
   self.doBackProp = true
   self.requiresGradients = true
   self.name = ''
   self.saveMem = false
   self.gpucompatible = false
end

function Module:parameters()
   if self.weight and self.bias then
      return {self.weight, self.bias}, {self.gradWeight, self.gradBias}
   elseif self.weight then
      return {self.weight}, {self.gradWeight}
   elseif self.bias then
      return {self.bias}, {self.gradBias}
   else
      return
   end
end

function Module:updateOutput(input)
   return self.output
end

function Module:forward(input)
   return self:updateOutput(input)
end

function Module:backward(input, gradOutput, scale)
   scale = scale or 1
   self:updateGradInput(input, gradOutput)
   self:accGradParameters(input, gradOutput, scale)
   return self.gradInput
end

function Module:updateGradInput(input, gradOutput)
   return self.gradInput
end

function Module:accGradParameters(input, gradOutput, scale)
end

function Module:zeroGradParameters()
   local _,gradParams = self:parameters()
   if gradParams then
      for i=1,#gradParams do
         gradParams[i]:zero()
      end
   end
   if self.modules then 
      for i=1,#self.modules do
        self.modules[i]:zeroGradParameters()
      end
   end
end

function Module:updateParameters()
   if self.modules then 
      for i=1,#self.modules do
         self.modules[i]:updateParameters()
      end
   end
end

function Module:clone(...)
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   if select('#',...) > 0 then
      clone:share(self,...)
   end
   return clone
end

function Module:type(type)
   -- find all tensors and convert them
   for key,param in pairs(self) do
      if torch.typename(param) and torch.typename(param):find('torch%..+Tensor') then
         self[key] = param:type(type)
      end
   end
   -- find submodules in classic containers 'modules'
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:type(type)
      end
   end
   return self
end

function Module:setTestMode(testbool)
   -- find submodules in classic containers 'modules'
   if self.testmode~=nil then self.testmode=testbool end
   if self.modules then
      for _,module in ipairs(self.modules) do
         if module.setTestMode then module:setTestMode(testbool) end
      end
   end
   return self
end

function Module:setInPlace(inplacebool)
   -- find submodules in classic containers 'modules'
   if self.inplace~=nil then self.inplace=inplacebool end
   if self.modules then
      for _,module in ipairs(self.modules) do
         if module.setInPlace then module:setInPlace(inplacebool) end
      end
   end
   return self
end

function Module:clipWeights(normbound)
   -- find submodules in classic containers 'modules'
   if self.modules then
      for _,module in ipairs(self.modules) do
         if module.clipWeights then module:clipWeights(normbound) end
      end
   end
   return self
end

local function zapTensor(a)
   if a then 
      a:resize(0)
      if a:storage() then
         a:storage():resize(0) 
      end
   end
end

function Module:getDisposableTensors()
   return {self.output, self.gradInput}
end

function Module:clean()
   if self.modules then
      for _,module in ipairs(self.modules) do
         if module.clean then module:clean() end
      end
   end
   local DT=self:getDisposableTensors()
   for _,a in ipairs(DT) do
      zapTensor(a)
   end   
end

function Module:float()
   return self:type('torch.FloatTensor')
end

function Module:double()
   return self:type('torch.DoubleTensor')
end

function Module:cuda()
   return self:type('torch.CudaTensor')
end

function Module:reset()
end

function Module:__call__(input, gradOutput)
   self:forward(input)
   if gradOutput then
      self:backward(input, gradOutput)
      return self.output, self.gradInput
   else
      return self.output
   end
end

function Module:setBackProp(BPbool)
   -- returns whether the module needs gradients or not
   self.doBackProp = BPbool or false
   self.requiresGradients=self:needGradients()
   return self.requiresGradients or self.doBackProp
end

function Module:needGradients()
   return false
end

function Module:setLearningRate(...)
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:setLearningRate(...) 
      end
   end
end

function Module:setSaveMem(...)
   self.saveMem=...
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:setSaveMem(...) 
      end
   end
end

function Module:setMomentum(...)
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:setMomentum(...) 
      end
   end
end

function Module:setWeightDecay(...)
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:setWeightDecay(...) 
      end
   end
end


function Module:autoLR(...)
   if self.modules then
      for _,module in ipairs(self.modules) do
         module:autoLR(...) 
      end
   end
end

      
function Module:getByName(name)
   if self.name==name then 
      return self 
   end 
   if self.modules then
      local mod
      local count=0
      for idx=1,#self.modules do
         if self.modules[idx]:getByName(name) then
            count=count+1
            mod = self.modules[idx]:getByName(name)
         end 
      end
      if count==1 then return mod end
      if count==0 then return end
      if count >1 then error('error : many layers with name '..name) end
   end
end

function Module:setName(name)
   self.name=name
end

function Module:isGPUCompatible(...)
   local gpucompatible = self.gpucompatible
   if self.modules then
      for _,module in ipairs(self.modules) do
         gpucompatible = gpucompatible and module:isGPUCompatible()
      end
   end
   return gpucompatible
end



