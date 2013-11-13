local ReLU = torch.class('nn.ReLU', 'nn.Module')

function ReLU:updateOutput(input)
   return input.nn.ReLU_updateOutput(self, input)
end

function ReLU:updateGradInput(input, gradOutput)
   return input.nn.ReLU_updateGradInput(self, input, gradOutput)
end
