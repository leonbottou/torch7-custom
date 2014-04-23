local MultiClassNLLCriterion, parent = torch.class('nxn.MultiClassNLLCriterion', 'nxn.Criterion')

function MultiClassNLLCriterion:__init()
   parent.__init(self)
   self.tmp=torch.Tensor()
end

function MultiClassNLLCriterion:updateOutput(input, target)
   self.tmp:resizeAs(input)
   self.tmp:copy(input)
   
   self.output=self.tmp:cmul(target):mul(-1):exp():add(1):log():sum()
   return self.output
end


function MultiClassNLLCriterion:updateGradInput(input, target)
   self.gradInput:resizeAs(input)
   self.gradInput:zero()

   self.gradInput:map2(input, target, function(x, inp, tgt)  return 1/(1+math.exp(-1*inp*tgt))*(-1*tgt)*math.exp(-1*inp*tgt) end)

   return self.gradInput
end
