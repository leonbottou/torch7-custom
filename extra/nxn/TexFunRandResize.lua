local TexFunRandResize, parent = torch.class('nxn.TexFunRandResize', 'nxn.ExtractInterpolate')

function TexFunRandResize:__init(scale1, scale2, testscale)
   parent.__init(self)
   self:setScales(scale1, scale2, testscale)
end

function TexFunRandResize:setScales(scale1, scale2, testscale)
   if not scale1 and scale2 and testscale then
      error('TexFunRandResize(scale1, scale2, testscale) or TexFunRandResize:setScales(scale1, scale2, testscale)')
   end
   self.scale1=scale1
   self.scale2=scale2
   self.testscale=testscale
end

function TexFunRandResize:updateOutput(input)
   
   if self.testmode then 
      self.scale=testscale 
   else
      self.scale=torch.uniform(self.scale1,self.scale2)
   end
   
   
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


function TexFunRandResize:updateGradInput(input, gradOutput)
   return 
end
