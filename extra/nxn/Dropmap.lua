local Dropmap, parent = torch.class('nxn.Dropmap', 'nxn.Module')

function Dropmap:__init(p)
   parent.__init(self)
   self.p = p
   self.mask=torch.Tensor()
   self.testmode=false
   self.sameoverbatch=1
   if (not p) or p<0 or p>1 then
      error('nxn.Dropmap(0<p<1), p = drop probability (p=0 => everything goes through)')
   end
end

function Dropmap:updateOutput(input)
   if not self.testmode then
      if self.sameoverbatch then
         self.mask:resize(input:size(input:dim())):bernoulli(1-self.p)
      else
         self.mask:resize(input:size(input:dim())*input:size(1)):bernoulli(1-self.p)
      end
      input.nxn.Dropmap_updateOutput(self, input)
   else 
      self.output:copy(input):mul(1-self.p)
   end
   return self.output
end

function Dropmap:updateGradInput(input, gradOutput)
   if not self.testmode then
      input.nxn.Dropmap_updateGradInput(self, input, gradOutput)
   else
      error('cannot backprop through Dropmap in test mode...')
   end
   return self.gradInput
end
