local Resize, parent = torch.class('nxn.Resize', 'nxn.Module')

function Resize:__init(scale1, scale2, testscale)
   parent.__init(self)
   self.scale1=scale1 or 1
   self.scale2=scale2
   self.testscale=testscale or self.scale1
   self.tmp=torch.Tensor()
   self.testmode=false
   self.gradInput=nil
end

function Resize:updateOutput(input)
   if self.scale2 then
      self:setScale(torch.uniform(self.scale1,self.scale2))
   else
      self:setScale(self.scale1)
   end

   if self.testmode then
      self:setScale(self.testscale)
   end

   if self.scale==1 then
      return input
   end

   if input:type() ~= 'torch.CudaTensor' then
      require 'image'
      self.output=image.scale(input:transpose(3,4):transpose(2,3):contiguous():resize((input:size(4)*input:size(1)), input:size(2), input:size(3)), input:size(3)*self.scale, input:size(2)*self.scale)
      self.output:resize(input:size(1), input:size(4), self.output:size(2), self.output:size(3))
      self.output=self.output:transpose(2,3):transpose(3,4):contiguous()
   else
      input.nxn.Resize_updateOutput(self, input)
   end
   return self.output
end

function Resize:setScale(scale)
   self.scale=scale or 1
end

function nxn.Resize:setScales(scale1,scale2, testscale)
   self.scale1=scale1 or 1
   self.scale2=scale2
   self.testscale=testscale or self.scale1
end


function Resize:updateGradInput(input, gradOutput)
   return 
end
