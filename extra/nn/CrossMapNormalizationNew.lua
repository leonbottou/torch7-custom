local CrossMapNormalizationNew, parent = torch.class('nn.CrossMapNormalizationNew', 'nn.Module')

function CrossMapNormalizationNew:__init(alpha, beta, k, n)
   parent.__init(self)
   self.alpha = alpha or 1e-4 
   self.beta = beta or 0.75
   self.k = k or 1
   self.n = n or 5
   self.z=torch.Tensor()
end

function CrossMapNormalizationNew:updateOutput(input)
   input.nn.CrossMapNormalizationNew_updateOutput(self, input)
   return self.output
end

function CrossMapNormalizationNew:updateGradInput(input, gradOutput)
   input.nn.CrossMapNormalizationNew_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
