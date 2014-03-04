local ReLU, parent = torch.class('nxn.ReLU', 'nxn.Module')

function ReLU:__init()
   parent.__init(self)
   self.inplace=0
end

function ReLU:updateOutput(input)
   if self.inplace==1 then
      self.output=input
   end
   return input.nxn.ReLU_updateOutput(self, input)
end

function ReLU:updateGradInput(input, gradOutput)
   if self.inplace==1 then
      self.gradInput=gradOutput
   end
   return input.nxn.ReLU_updateGradInput(self, input, gradOutput)
end
