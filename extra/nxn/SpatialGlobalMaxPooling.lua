local SpatialGlobalMaxPooling, parent = torch.class('nxn.SpatialGlobalMaxPooling', 'nxn.Module')

local help_str = 
[[This is the global max-pooling module.
It performs a global max-pooling on each feature map and returns a batchsize*1*1*channels output.

Usage : m = nxn.SpatialGlobalMaxPooling(poolW, poolH, dW, dH)

It only works in BATCH MODE (4D) :
- with the following input layout : (batch, y, x, channels).
- channels are the contiguous dimension.
- a single image must be a (1, y, x, channels) tensor.

The module doesn't require fixed-size inputs.]]

function SpatialGlobalMaxPooling:__init()
   parent.__init(self)

   self.indices=torch.Tensor()
   self.gpucompatible = true
end


function SpatialGlobalMaxPooling:updateOutput(input)
   input.nxn.SpatialGlobalMaxPooling_updateOutput(self, input)
   return self.output
end

function SpatialGlobalMaxPooling:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nxn.SpatialGlobalMaxPooling_updateGradInput(self, input, gradOutput)
      return self.gradInput
   end
end

function SpatialGlobalMaxPooling:getDisposableTensors()
   return {self.output, self.gradInput, self.indices}
end
