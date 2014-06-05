local SpatialMaxPooling, parent = torch.class('nxn.SpatialMaxPooling', 'nxn.Module')

function SpatialMaxPooling:__init(poolW, poolH, dW, dH, shdmem)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.poolW = poolW
   self.poolH = poolH
   self.dW = dW
   self.dH = dH
   self.shdmem = shdmem or 1
   self.indices = torch.Tensor()
   self.gpucompatible = true
end


function SpatialMaxPooling:updateOutput(input)
   input.nxn.SpatialMaxPooling_updateOutput(self, input)
   return self.output
end

function SpatialMaxPooling:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nxn.SpatialMaxPooling_updateGradInput(self, input, gradOutput)
      return self.gradInput
   end
end

