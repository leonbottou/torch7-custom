local TexFunCropJitter, parent = torch.class('nxn.TexFunCropJitter', 'nxn.ExtractInterpolate')

function TexFunCropJitter:__init(cropx, cropy)
   parent.__init(self)
   self:setCrops(cropx, cropy)

end

function TexFunCropJitter:setCrops(cropx, cropy)
   if not cropx then
      error('TexFunFixedResize:setCrops(cropx[, cropy]) or TexFunFixedResize(cropx[, cropy])')
   end
   self.cropx=cropx
   self.cropy=cropy or cropx
end

function TexFunCropJitter:updateOutput(input)
   local xcrop, ycrop
   if self.testmode then
      xcrop = math.floor(self.cropx/2)
      ycrop = math.floor(self.cropy/2)
   else
      xcrop = math.random(0,self.cropx)
      ycrop = math.random(0,self.cropy)
   end


   if input:type() == 'torch.CudaTensor' then
      local x1=xcrop
      local y1=ycrop
      
      local x2=input:size(3)-self.cropx+xcrop
      local y2=ycrop
      
      local x3=input:size(3)-self.cropx+xcrop
      local y3=input:size(2)-self.cropy+ycrop

      local x4=xcrop
      local y4=input:size(2)-self.cropy+ycrop

      local targety = input:size(2)-self.cropy
      local targetx = input:size(3)-self.cropx
      
      self:updateOutputCall(input, targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4)
   else
   -- y u no gpu
      self.xcrop=self.cropx
      self.ycrop=self.cropy
      self.xstart=xcrop+1
      self.ystart=ycrop+1
      self.randflip=0 -- this is a residual of CPU nxn.Jitter()
      input.nxn.Jitter_updateOutput(self, input)      
   end   
   return self.output
end




function TexFunCropJitter:updateGradInput(input, gradOutput)
   return 
end
