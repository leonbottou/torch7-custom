local Sequential, parent = torch.class('nxn.Sequential', 'nxn.Module')

function Sequential:__init()
   self.modules = {}
end

function Sequential:add(module)
   if #self.modules == 0 then
      self.gradInput = module.gradInput
   end
   table.insert(self.modules, module)
   self.output = module.output
   return self
end

function Sequential:size()
   return #self.modules
end

function Sequential:get(index)
   return self.modules[index]
end

function Sequential:updateOutput(input)
   local currentOutput = input
   for i=1,#self.modules do 
      currentOutput = self.modules[i]:updateOutput(currentOutput)
   end 
   self.output = currentOutput
   return currentOutput
end

function Sequential:updateGradInput(input, gradOutput)
   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      if currentModule.requiresGradients or currentModule.doBackProp then
         currentGradOutput = currentModule:updateGradInput(previousModule.output, currentGradOutput)
      end
      currentModule = previousModule
   end
   if currentModule.requiresGradients or currentModule.doBackProp then
      currentGradOutput = currentModule:updateGradInput(input, currentGradOutput)
   end
   self.gradInput = currentGradOutput
   return currentGradOutput
end

function Sequential:accGradParameters(input, gradOutput, scale)
   scale = scale or 1

   local currentGradOutput = gradOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accGradParameters(previousModule.output, currentGradOutput, scale)
      if currentModule.requiresGradients or currentModule.doBackProp then
         currentGradOutput = currentModule.gradInput
      end
      currentModule = previousModule
   end
   
   if currentModule.requiresGradients or currentModule.doBackProp then
      currentModule:accGradParameters(input, currentGradOutput, scale)
   end
end

function Sequential:reset(stdv)
   for i=1,#self.modules do
      self.modules[i]:reset(stdv)
   end
end

function Sequential:parameters()
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

function Sequential:__tostring__()
   local tab = '     |  '
   local line = '\n'
   local next = ' -> '
   local str = 'nxn.Sequential'
   str = str .. ' {' .. line .. tab .. '[input'
   for i=1,#self.modules do
      str = str .. next .. '(' .. i .. ')'
   end
   str = str .. next .. 'output]'
   for i=1,#self.modules do
      str = str .. line .. tab .. '(' .. i .. '): ' .. tostring(self.modules[i]):gsub(line, line .. tab)
   end
   str = str .. line .. '}'
   return str
end

function Sequential:needGradients()
   -- container needs the gradients if one of the modules does
   local switch=false
   for i=1,#self.modules do 
      switch = switch or self.modules[i]:needGradients()
   end 
   return switch
end

function Sequential:setBackProp(BPbool)
   self.doBackProp=false or BPbool
   self.requiresGradients=self:needGradients()
   local switch=self.doBackProp
   for i=1,#self.modules do 
      switch = self.modules[i]:setBackProp(switch)
   end 
   return self.requiresGradients or self.doBackProp
end
