local Column, parent = torch.class('nxn.Column', 'nxn.Module')

-- this module splits the input along the innermost dimension and concatenates along the innermost dimension in the end
-- nxn.Column(torch.LongStorage({split1, split2, split3, ...}))

function Column:__init(splits)
   self.modules = {}
   self.inputs  = {}
   self.gradOutputs = {}
   self.splitSizes = splits
   
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()

end

function Column:add(module)
   table.insert(self.modules, module)
   return self
end

function Column:size()
   return #self.modules
end

function Column:get(index)
   return self.modules[index]
end

function Column:updateOutput(input)
   local first=1
   for i=1,#self.splitSizes do
      if not self.inputs[i] then
         self.inputs[i]=input.new()
      end
      self.inputs[i]:typeAs(input)
      self.inputs[i]:resizeAs(input:narrow(input:dim(), first, self.splitSizes[i]))
      self.inputs[i]:copy(input:narrow(input:dim(), first, self.splitSizes[i]))
      first=first+self.splitSizes[i]
   end
   
   local outdims=0
   for i=1,#self.modules do 
      self.modules[i]:updateOutput(self.inputs[i])
   end 
   
   for i=1,#self.modules do 
      outdims = outdims + self.modules[i].output:size(self.modules[i].output:dim())
   end 
   
   local outputsize=self.modules[1].output:size()
   outputsize[outputsize:size()]=outdims
   self.output:resize(outputsize)
   
   local outfirst=1
   for i=1,#self.modules do 
      local outsplitsize=self.modules[i].output:size(self.modules[i].output:dim())
      t=self.output:narrow(self.modules[i].output:dim(), outfirst, outsplitsize)
      t:copy(self.modules[i].output)
      outfirst=outfirst+outsplitsize
   end 

   return self.output
end

function Column:updateGradInput(input, gradOutput)
   local first=1

   if gradOutput then    
      for i=1,#self.modules do 
         if self.modules[i].requiresGradients or self.modules[i].doBackProp then
            if not self.gradOutputs[i] then
               self.gradOutputs[i]=input.new()
            end
            local outsplitsize=self.modules[i].output:size(self.modules[i].output:dim())
            self.gradOutputs[i]:resizeAs(gradOutput:narrow(gradOutput:dim(), first, outsplitsize))
            self.gradOutputs[i]:copy(gradOutput:narrow(gradOutput:dim(), first, outsplitsize))
            first=first + outsplitsize
            self.modules[i]:updateGradInput(self.inputs[i], self.gradOutputs[i])
         end
      end 
   end

   if self.doBackProp then   
      self.gradInput:resizeAs(input)
      
      first=1
      
      for i=1,#self.splitSizes do
         tgradInput = self.gradInput:narrow(input:dim(), first, self.splitSizes[i])
         tgradInput:copy(self.modules[i].gradInput)
         first=first+self.splitSizes[i]
      end   
      
      return self.gradInput
   end
end

function Column:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   if gradOutput then    
      for i=1,#self.modules do 
         if self.modules[i].requiresGradients or self.modules[i].doBackProp then
            self.modules[i]:accGradParameters(self.inputs[i], self.gradOutputs[i], scale)
         end
      end   
   end
end

function Column:reset(stdv)
   for i=1,#self.modules do
      self.modules[i]:reset(stdv)
   end
end

function Column:getDisposableTensors()
   local tbl={self.output, self.gradInput}
   for key,param in pairs(self.inputs) do
      table.insert(tbl, param)
   end
   for key,param in pairs(self.gradOutputs) do
      table.insert(tbl, param)
   end
   return tbl
end

function Column:type(type)
   -- find all tensors and convert them
   for key,param in pairs(self) do
      if torch.typename(param) and torch.typename(param):find('torch%..+Tensor') then
         self[key] = param:type(type)
      end
   end
   for key,param in pairs(self.inputs) do
      if torch.typename(param) and torch.typename(param):find('torch%..+Tensor') then
         self.inputs[key] = param:type(type)
      end
   end
   for key,param in pairs(self.gradOutputs) do
      if torch.typename(param) and torch.typename(param):find('torch%..+Tensor') then
         self.gradOutputs[key] = param:type(type)
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

function Column:parameters()
   local function tinsert(to, from)
      if type(from) == 'table' then
         for i=1,#from do
            tinsert(to,from[i])
         end
      else
         table.insert(to,from)
      end
   end
   local w = {}
   local gw = {}
   for i=1,#self.modules do
      local mw,mgw = self.modules[i]:parameters()
      if mw then
         tinsert(w,mw)
         tinsert(gw,mgw)
      end
   end
   return w,gw
end

function Column:__tostring__()
   local tab = '     '
   local line = '\n'
   local next = ' -> '
   local str = 'nxn.Column'
   str = str .. ' {'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end


function Column:needGradients()
   -- container needs the gradients if one of the modules does
   local switch=false
   for i=1,#self.modules do 
      switch = switch or self.modules[i]:needGradients()
   end 
   return switch
end

function Column:setBackProp(BPbool)
   self.doBackProp=false or BPbool
   self.requiresGradients=self:needGradients()
   for i=1,#self.modules do 
      self.modules[i]:setBackProp(self.doBackProp)
   end 
   return self.requiresGradients or self.doBackProp
end
