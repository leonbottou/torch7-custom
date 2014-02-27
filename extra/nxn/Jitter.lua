local Jitter, parent = torch.class('nxn.Jitter', 'nxn.Module')

function Jitter:__init(xcrop, ycrop, flip)
   parent.__init(self)
   self.xcrop=xcrop or 0
   self.ycrop=ycrop or 0
   self.testmode=false;
   if flip==true or flip==1 then 
      self.flip=1 
   else 
      self.flip=0 
   end
end

function Jitter:updateOutput(input)
   self.xstart=math.random(1,self.xcrop)
   self.ystart=math.random(1,self.ycrop)
   self.randflip=torch.bernoulli(0.5)
   local out=input:narrow(3, xstart, input:size(3)-self.xcrop):narrow(2, xstart, input:size(2)-self.ycrop):contiguous()
   
   if self.flip==0 then
      self.output=out
   else 
      self.output:resizeAs(out)
      for xx=1,out:size(3) do
         self.output:select(3,xx):copy(out:select(3, out:size(3)+1-xx))
      end
   end
   return self.output
end

function Jitter:updateOutput(input)
   
   if self.testmode then
      self.xstart=math.floor(self.xcrop/2)
      self.ystart=math.floor(self.ycrop/2)
      self.randflip=0
   else
      self.xstart=math.random(1,self.xcrop)
      self.ystart=math.random(1,self.ycrop)
      self.randflip=torch.bernoulli(0.5)
   end
   
   input.nxn.Jitter_updateOutput(self, input)

   return self.output
end

function Jitter:updateGradInput(input, gradOutput)
   if self.gradInput then
   
   end
end
