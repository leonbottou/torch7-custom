local ReLU = torch.class('nxn.ReLU', 'nxn.Module')

function ReLU:updateOutput(input)
   return input.nxn.ReLU_updateOutput(self, input)
end

function ReLU:updateGradInput(input, gradOutput)
   return input.nxn.ReLU_updateGradInput(self, input, gradOutput)
end
