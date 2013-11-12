local CrossMapNormalization, parent = torch.class('nn.CrossMapNormalization', 'nn.Module')

function CrossMapNormalization:__init(alpha, beta, k, n, dimension)
   parent.__init(self)
   self.dimension = dimension or 1
   self.alpha = alpha or 1e-4 
   self.beta = beta or 0.75
   self.k = k or 1
   self.n = n or 5
end

function CrossMapNormalization:updateOutput(input)
   input.nn.CrossMapNormalization_updateOutput(self, input)
   return self.output
end

function CrossMapNormalization:updateGradInput(input, gradOutput)
   input.nn.CrossMapNormalization_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
