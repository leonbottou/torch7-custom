local SpatialGlobalMaxPoolingNew, parent = torch.class('nn.SpatialGlobalMaxPoolingNew', 'nn.Module')

function SpatialGlobalMaxPoolingNew:__init()
   parent.__init(self)

   self.indices=torch.Tensor()
end


function SpatialGlobalMaxPoolingNew:updateOutput(input)
   input.nn.SpatialGlobalMaxPoolingNew_updateOutput(self, input)
   return self.output
end

function SpatialGlobalMaxPoolingNew:updateGradInput(input, gradOutput)
   if self.gradInput then
      input.nn.SpatialGlobalMaxPoolingNew_updateGradInput(self, input, gradOutput)
      return self.gradInput
   end
end

