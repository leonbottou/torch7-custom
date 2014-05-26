require('torch')
require('libnxn')

include('Module.lua')
include('Sequential.lua')
include('Column.lua')
--include('ConvLayer.lua')
include('SpatialConvolution.lua')
include('Copy.lua')
include('Reshape.lua')

include('ReLU.lua')
include('Affine.lua')
include('Dropout.lua')
include('Dropmap.lua')
include('SpatialGlobalMaxPooling.lua')
include('SpatialMaxPooling.lua')
include('SoftMax.lua')
include('CrossMapNormalization.lua')

include('Criterion.lua')
include('ClassNLLCriterion.lua')
include('MultiClassNLLCriterion.lua')
include('LogSoftMax.lua')

include('Jitter.lua')
include('Resize.lua')
include('ExtractInterpolate.lua')

include('NeuralNet.lua')
include('Dataset.lua')



if false then 

include('LargeDataset.lua')


include('Linear.lua')

include('ConvProto.lua')
end

include('testSgemm.lua')

