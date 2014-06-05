local TexFunFixedResize, parent = torch.class('nxn.TexFunFixedResize', 'nxn.ExtractInterpolate')

function TexFunFixedResize:__init(scale)
   parent.__init(self)
   self.scale=scale
end

function TexFunFixedResize:setScale(scale)
   if not scale then
      error('TexFunFixedResize:setScale(scale) or TexFunFixedResize(scale)')
   end
   self.scale=scale or 1
end

function TexFunFixedResize:updateOutput(input)

   if input:type() == 'torch.CudaTensor' then
      local x1=1
      local y1=1
      
      local x2=input:size(3)
      local y2=1
      
      local x3=input:size(3)
      local y3=input:size(2)

      local x4=1
      local y4=input:size(2)

      local targety = input:size(2)*self.scale
      local targetx = input:size(3)*self.scale
      
      self:updateOutputCall(input, targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4)
   else
   -- y u no gpu
         require 'image'
         self.output=image.scale(input:transpose(3,4):transpose(2,3):contiguous():resize((input:size(4)*input:size(1)), input:size(2), input:size(3)), input:size(3)*self.scale, input:size(2)*self.scale)
         self.output:resize(input:size(1), input:size(4), self.output:size(2), self.output:size(3))
         self.output=self.output:transpose(2,3):transpose(3,4):contiguous()
   end   
   return self.output
end




function TexFunFixedResize:updateGradInput(input, gradOutput)
   return 
end
