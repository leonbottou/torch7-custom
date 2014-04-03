local Resize, parent = torch.class('nxn.Resize', 'nxn.Module')

function Resize:__init(scale)
   parent.__init(self)
   self.scale=scale or 2
   self.tmp=torch.Tensor()
   self.tmp2=torch.Tensor()
   self.gradInput=nil
end

function Resize:updateOutput(input)
   return input.nxn.Resize_updateOutput(self, input)
end

function Resize:updateGradInput(input, gradOutput)
   return 
end
