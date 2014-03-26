local LogSoftMax = torch.class('nxn.LogSoftMax', 'nxn.Module')

function LogSoftMax:updateOutput(input)
   return input.nxn.LogSoftMax_updateOutput(self, input)
end

function LogSoftMax:updateGradInput(input, gradOutput)
   return input.nxn.LogSoftMax_updateGradInput(self, input, gradOutput)
end
