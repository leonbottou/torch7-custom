local Affine, parent = torch.class('nn.Affine', 'nn.Module')

function Affine:__init(factor, offset)
   parent.__init(self)
   self.factor = factor
   self.offset = offset or 0
   if not factor then
      error('nn.Affine(factor [, offset])')
   end
end

function Affine:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   self.output:mul(self.factor)
   self.output:add(self.offset)
   return self.output
end

function Affine:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(input):copy(gradOutput):mul(self.factor)
   return self.gradInput
end
