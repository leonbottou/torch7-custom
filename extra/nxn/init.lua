require('torch')
require('libnxn')

include('Module.lua')
include('Sequential.lua')
include('Column.lua')
include('SpatialConvolution.lua')
include('SpatialConvolutionUnfold.lua')
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
include('TexFunCropJitter.lua')
include('TexFunCustom.lua')
include('TexFunFixedResize.lua')
include('TexFunRandFlip.lua')
include('TexFunRandResize.lua')

include('NeuralNet.lua')
include('Dataset.lua')


include('testSgemm.lua')

