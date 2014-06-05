local CrossMapNormalization, parent = torch.class('nxn.CrossMapNormalization', 'nxn.Module')

function CrossMapNormalization:__init(alpha, beta, k, n)
   parent.__init(self)
   self.alpha = alpha or 1e-4 
   self.beta = beta or 0.75
   self.k = k or 1
   self.n = n or 5
   self.z=torch.Tensor()
   self.gpucompatible = true
end

function CrossMapNormalization:updateOutput(input)
   input.nxn.CrossMapNormalization_updateOutput(self, input)
   return self.output
end

function CrossMapNormalization:updateGradInput(input, gradOutput)
   input.nxn.CrossMapNormalization_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function CrossMapNormalization:getDisposableTensors()
   return {self.output, self.gradInput, self.z}
end
