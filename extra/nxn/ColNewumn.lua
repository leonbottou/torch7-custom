local ColNewumn, parent = torch.class('nxn.ColNewumn', 'nxn.Module')

-- this module splits the input along the innermost dimension and concatenates along the innermost dimension in the end
-- input split compatible only with nxn.SpatialConvolution modules (must be alone or first in a nxn.Sequential container)
-- gradOutput split : same (must be last)

function ColNewumn:__init(splits)
   self.modules = {}
   self.inputs  = {}
   self.gradOutputs = {}
   self.splitSizes = splits
   
   self.output = torch.Tensor()
   self.gradInput = torch.Tensor()

end

function ColNewumn:add(module)
   table.insert(self.modules, module)
   return self
end

function ColNewumn:size()
   return #self.modules
end

function ColNewumn:get(index)
   return self.modules[index]
end

function ColNewumn:updateOutput(input)
-- outputs do not overlap, but inputs do
   local regen=false
   for i=1,#self.modules do 
      if not self.inputs[i] then
         self.inputs[i]=input[{{},{},{},self.splitSizes[i]}]
      end
      self.modules[i]:updateOutput(self.inputs[i])
      if self.modules[i].output:storage() ~= self.output:storage() then
         regen=true
      end
   end 
   
   local outdims=0
   for i=1,#self.modules do 
      outdims = outdims + self.modules[i].output:size(self.modules[i].output:dim())
   end 
   
   local outputsize=self.modules[1].output:size()
   outputsize[outputsize:size()]=outdims
   self.output:resize(outputsize)
   
   local outfirst=1
   if regen==true then
      for i=1,#self.modules do 
         local outsplitsize=self.modules[i].output:size(self.modules[i].output:dim())
         t=self.output:narrow(self.modules[i].output:dim(), outfirst, outsplitsize)
         t:copy(self.modules[i].output)
         self.modules[i].output=self.output:narrow(self.modules[i].output:dim(), outfirst, outsplitsize)
         outfirst=outfirst+outsplitsize
      end 
   end
   return self.output
end

function ColNewumn:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):zero()
   local first=1
      
   for i=1,#self.modules do 
      local outsplitsize=self.modules[i].output:size(self.modules[i].output:dim())
      self.gradOutputs[i]=(gradOutput:narrow(gradOutput:dim(), first, outsplitsize))
      first=first + outsplitsize
      self.modules[i].gradInput=self.gradInput[{{},{},{},self.splitSizes[i]}]
      self.modules[i]:setAddGrads(1)
      self.modules[i]:updateGradInput(self.inputs[i], self.gradOutputs[i])
   end 
  
   return self.gradInput
end

function ColNewumn:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   
   for i=1,#self.modules do 
      self.modules[i]:accGradParameters(self.inputs[i], self.gradOutputs[i], scale)
   end   

end

function ColNewumn:accUpdateGradParameters(input, gradOutput, lr)
   
   for i=1,#self.modules do 
      self.modules[i]:accUpdateGradParameters(self.inputs[i], self.gradOutputs[i], lr)
   end   

end

function ColNewumn:zeroGradParameters()
  for i=1,#self.modules do
     self.modules[i]:zeroGradParameters()
  end
end

function ColNewumn:updateParameters(learningRate)
   for i=1,#self.modules do
      self.modules[i]:updateParameters(learningRate)
   end
end

function ColNewumn:share(mlp,...)
   for i=1,#self.modules do
      self.modules[i]:share(mlp.modules[i],...); 
   end
end

function ColNewumn:reset(stdv)
   for i=1,#self.modules do
      self.modules[i]:reset(stdv)
   end
end

function ColNewumn:parameters()
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

function ColNewumn:__tostring__()
   local tab = '  '
   local line = '\n'
   local next = ' -> '
   local str = 'nxn.ColNewumn'
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
