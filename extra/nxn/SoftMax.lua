local SoftMax, parent = torch.class('nxn.SoftMax', 'nxn.Module')

function SoftMax:__init()
   parent.__init(self)
   self.gpucompatible = true
end

function nxn.SoftMax:updateOutput(input)
   self.output:resizeAs(input)
   self.output:copy(input)
   local dim=#(#self.output)
   local maxvalues=self.output:max(dim)
   maxvalues=maxvalues:expandAs(self.output)
   self.output:add(-1,maxvalues)
   self.output:exp()
   local sumvalues=self.output:sum(dim)
   sumvalues=sumvalues:expandAs(self.output)
   self.output:cdiv(sumvalues)
   return self.output
end

function nxn.SoftMax:updateGradInput(input, gradOutput)
   local dim=#(#self.output)
   self.gradInput:resizeAs(gradOutput)
   self.gradInput:copy(gradOutput)
   self.gradInput:cmul(self.output)
   local tmp=self.gradInput:sum(dim)
   tmp=tmp:expandAs(self.gradInput)
   self.gradInput:addcmul(-1,self.output,tmp)
   return self.gradInput
end
