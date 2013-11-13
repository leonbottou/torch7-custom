local Affine, parent = torch.class('nn.Affine', 'nn.Module')

function Affine:__init(factor)
   parent.__init(self)
   self.factor = factor
   if not factor then
      error('nn.Affine(factor)')
   end
end

function Affine:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:mul(self.factor)
   return self.output
end

function Affine:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):copy(gradOutput):mul(self.factor)
   return self.gradInput
end
