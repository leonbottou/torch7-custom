local Dropmap, parent = torch.class('nxn.Dropmap', 'nxn.Module')

function Dropmap:__init(p)
   parent.__init(self)
   self.p = p
   self.mask=torch.Tensor()
   self.testmode=false
   self.sameoverbatch=1
   self.inplace=0
   self.outputSave=self.output
   self.gradInputSave=self.gradInput
   if (not p) or p<0 or p>1 then
      error('nxn.Dropmap(0<p<1), p = drop probability (p=0 => everything goes through)')
   end
   self.gpucompatible = true
end

function Dropmap:updateOutput(input)
   if not self.testmode then
      if self.inplace==1 then
         self.output=input
      else
         self.output=self.outputsave
      end
      if self.sameoverbatch==1 then
         self.mask:resize(input:size(input:dim())):bernoulli(1-self.p)
      else
         self.mask:resize(input:size(input:dim())*input:size(1)):bernoulli(1-self.p)
      end
      input.nxn.Dropmap_updateOutput(self, input)
   else 
      if self.inplace==1 then
         self.output=input
      else
         self.output=self.outputsave
         self.output:resizeAs(input):copy(input)
      end
      self.output:mul(1-self.p)
   end
   return self.output
end

function Dropmap:updateGradInput(input, gradOutput)
   if not self.testmode then
      if self.inplace==1 then
         self.gradInput=gradOutput
      else
         self.gradInput=self.gradInputSave
      end
      input.nxn.Dropmap_updateGradInput(self, input, gradOutput)
   else
      error('cannot backprop through Dropmap in test mode...')
   end
   return self.gradInput
end

function Dropmap:getDisposableTensors()
   return {self.output, self.gradInput, self.gradInputSave, self.outputSave, self.mask}
end
