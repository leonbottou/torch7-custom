local Dropout, parent = torch.class('nxn.Dropout', 'nxn.Module')

function Dropout:__init(p)
   parent.__init(self)
   self.p = p
   self.mask=torch.Tensor()
   self.testmode=false
   self.inplace=0
   if (not p) or p<0 or p>1 then
      error('nxn.Dropout(0<p<1), p = drop probability (p=0 => everything goes through)')
   end
end

function Dropout:updateOutput(input)
   if self.inplace==1 then
      self.output=input
   else
      self.output:resizeAs(input):copy(input)
   end
   if not self.testmode then
      self.mask:resizeAs(input):bernoulli(1-self.p)
      self.output:cmul(self.mask)
   else 
      self.output:mul(1-self.p)
   end
   return self.output
end

function Dropout:updateGradInput(input, gradOutput)
   if not self.testmode then
      if self.inplace==1 then
         self.gradInput=gradOutput
      else
         self.gradInput:resizeAs(gradOutput):copy(gradOutput)
      end
      self.gradInput:cmul(self.mask)
   else
      error('cannot backprop through dropout in test mode...')
   end
   return self.gradInput
end
