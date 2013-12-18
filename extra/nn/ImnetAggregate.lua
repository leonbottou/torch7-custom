local ImnetAggregate, parent = torch.class('nn.ImnetAggregate', 'nn.Module')

function ImnetAggregate:__init()
   parent.__init(self)
   

   wbfile=torch.DiskFile('/ssd/t-maoqu/pythontorch/aggarray')
   data=wbfile:binary():readInt(40477)
   tmptensor=torch.IntTensor()
   tmptensor:set(data, 1, torch.LongStorage({40477}))
   self.aggtensor=tmptensor:float():clone()
   self.length=40477
   self.numclasses=21841
   self.chosentensor=torch.FloatTensor(torch.LongStorage({21841}))
   
end

function ImnetAggregate:updateOutput(input)
   input.nn.ImnetAggregate_updateOutput(self, input)
   return self.output
end

function ImnetAggregate:updateGradInput(input, gradOutput)
   input.nn.ImnetAggregate_updateGradInput(self, input, gradOutput)
   return self.gradInput
end
