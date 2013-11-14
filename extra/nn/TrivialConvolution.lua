local TrivialConvolution, parent = torch.class('nn.TrivialConvolution', 'nn.Module')

function TrivialConvolution:__init(inputSize, outputSize, dimension)
   parent.__init(self)

   self.weight = torch.Tensor(outputSize, inputSize)
   self.bias = torch.Tensor(outputSize)
   self.gradWeight = torch.Tensor(outputSize, inputSize)
   self.gradBias = torch.Tensor(outputSize)
   self.dimension = dimension or 1
   self:reset()
end

function TrivialConvolution:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end
end

function TrivialConvolution:updateOutput(input)
   -- check input size
   local osize = self.weight:size(1)
   local isize = self.weight:size(2)
   if input:size(self.dimension) ~= isize then
      error("TrivialConvolution: input tensor has wrong size")
   end
   local msize = input:nElement()/isize;
   -- transpose input
   local tinput = input:transpose(self.dimension,1)
   tinput = tinput:contiguous():resize(isize, msize)
   -- resize output
   self.output:resize(osize,msize)
   -- copy biases
   self.output:zero():addr(1, self.bias,input.new(msize):fill(1))
   -- do matrix product
   self.output:addmm(1, self.weight, tinput)
   -- resize output
   local dims=input:size()
   dims[self.dimension]=osize
   self.output:resize(dims)

   return self.output
end

function TrivialConvolution:updateGradInput(input, gradOutput)
   if self.gradInput then
      -- check input size
      local osize = self.weight:size(1)
      local isize = self.weight:size(2)
      local msize = input:nElement()/isize;

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      
      -- resize and transpose the stuff
      local gradinputdims=self.gradInput:size()
      self.gradInput = self.gradInput:transpose(self.dimension,1)      
      self.gradInput = self.gradInput:contiguous():resize(isize, msize)
      
      local gradoutputdims=gradoutput:size()
      gradOutput:resize(osize,msize)
      
      self.gradInput:addmm(0, 1, self.weight:transpose(1,2), gradOutput)
      
      -- resize the stuff back      
      gradOutput:resize(gradoutputdims)
      self.gradInput:resize(gradinputdims)
   
      return self.gradInput   
   end   
end

function TrivialConvolution:accGradParameters(input, gradOutput, scale)
   scale = scale or 1
   -- check input size
   local osize = self.weight:size(1)
   local isize = self.weight:size(2)
   local msize = input:nElement()/isize;
   
   -- resize and transpose the stuff
   local tinput = input:transpose(self.dimension,1)
   tinput = tinput:contiguous():resize(isize, msize)
   
   local gradoutputdims=gradoutput:size()
   gradOutput:resize(osize,msize)

   self.gradWeight:addmm(scale, gradOutput, tinput:t())
   self.gradBias:addmv(scale, gradOutput, tinput.new(msize):fill(1))
     
   -- resize the stuff back      
   gradOutput:resize(gradoutputdims) 
   
end

-- we do not need to accumulate parameters when sharing
TrivialConvolution.sharedAccUpdateGradParameters = TrivialConvolution.accUpdateGradParameters
