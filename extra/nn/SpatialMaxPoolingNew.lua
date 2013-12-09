local SpatialMaxPoolingNew, parent = torch.class('nn.SpatialMaxPoolingNew', 'nn.Module')

function SpatialMaxPoolingNew:__init(poolW, poolH, dW, dH, shdmem)
   parent.__init(self)

   dW = dW or 1
   dH = dH or 1

   self.poolW = poolW
   self.poolH = poolH
   self.dW = dW
   self.dH = dH
   self.shdmem = shdmem or 1
--   self.kslicestest = torch.Tensor()
end


function SpatialMaxPoolingNew:updateOutput(input)
   input.nn.SpatialMaxPoolingNew_updateOutput(self, input)
   return self.output
end

function SpatialMaxPoolingNew:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nn.SpatialMaxPoolingNew_updateGradInput(self, input, gradOutput)
      return self.gradInput
   end
end

