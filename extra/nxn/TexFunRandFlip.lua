local TexFunRandFlip, parent = torch.class('nxn.TexFunRandFlip', 'nxn.ExtractInterpolate')

function TexFunRandFlip:__init(flipprob)
   parent.__init(self)
   self:setFlipProb(flipprob)
   self.testmode=false
end

function TexFunRandFlip:setFlipProb(flipprob)
   self.flipprob=flipprob or 0.5
end

function TexFunRandFlip:updateOutput(input)
   local flip
   
   if self.testmode then 
      flip=0
   else
      flip=torch.bernoulli(self.flipprob)
   end
   
   if flip==0 then
      self.output:resizeAs(input):copy(input)
   else
      if input:type() == 'torch.CudaTensor' then
         local x1=input:size(3)
         local y1=1
         
         local x2=1
         local y2=1
         
         local x3=1
         local y3=input:size(2)

         local x4=input:size(3)
         local y4=input:size(2)

         local targety = input:size(2)
         local targetx = input:size(3)
         
         self:updateOutputCall(input, targety, targetx, y1, x1, y2, x2, y3, x3, y4, x4)
      else
         self.xcrop=0 -- this is a residual of CPU nxn.Jitter()
         self.ycrop=0 -- this is a residual of CPU nxn.Jitter()
         self.xstart=xcrop+1 -- this is a residual of CPU nxn.Jitter()
         self.ystart=ycrop+1 -- this is a residual of CPU nxn.Jitter()
         self.randflip=flip
         input.nxn.Jitter_updateOutput(self, input)      
      end
   end

   return self.output

end


function TexFunRandFlip:updateGradInput(input, gradOutput)
   return 
end
