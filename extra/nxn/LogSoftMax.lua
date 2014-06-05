local LogSoftMax = torch.class('nxn.LogSoftMax', 'nxn.Module')

function LogSoftMax:__init(p)
   parent.__init(self)
   self.gpucompatible = true
end

function LogSoftMax:updateOutput(input)
   input.nxn.LogSoftMax_updateOutput(self, input)
   return self.output
end

function LogSoftMax:updateGradInput(input, gradOutput)
   input.nxn.LogSoftMax_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
