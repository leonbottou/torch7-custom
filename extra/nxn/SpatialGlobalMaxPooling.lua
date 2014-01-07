local SpatialGlobalMaxPooling, parent = torch.class('nxn.SpatialGlobalMaxPooling', 'nxn.Module')

function SpatialGlobalMaxPooling:__init()
   parent.__init(self)

   self.indices=torch.Tensor()
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

