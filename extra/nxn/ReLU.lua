local ReLU, parent = torch.class('nxn.ReLU', 'nxn.Module')

function ReLU:__init()
   parent.__init(self)
   self.inplace=0
   self.outputSave=self.output
   self.gradInputSave=self.gradInput
   self.gpucompatible = true
end

function ReLU:updateOutput(input)
   if self.inplace==1 then
      self.output=input
   else
      self.output=self.outputSave
   end
   return input.nxn.ReLU_updateOutput(self, input)
end

function ReLU:updateGradInput(input, gradOutput)
   if self.inplace==1 then
      self.gradInput=gradOutput
   else
      self.gradInput=self.gradInputSave
   end
   return input.nxn.ReLU_updateGradInput(self, input, gradOutput)
end

function ReLU:getDisposableTensors()
   return {self.output, self.gradInput, self.gradInputSave, self.outputSave}
end
