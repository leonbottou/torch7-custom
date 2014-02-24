require 'optim'


local NeuralNet = torch.class('nxn.NeuralNet')

-- to do : 
-- parameter checks
-- weight visualization functions
-- train function
-- test function
-- set modules in test mode (dropout... => nxn.Module class)
-- put the LR and decay parameters in the nxn.Module class (each layer should have its own)
-- gradient clipping per-kernel

function NeuralNet:__init()
   self.network = {}
   self.criterion = {}
   self.meanoverset = torch.Tensor()
   self.datasetdir = ''
   self.trainset = {}
   self.testset = {}
   self.batchsize = 0
   
   self.checkpointdir = ''
   self.checkpointname = ''
   
   self.batchshuffle = {}
   
   self.momentum = 0
   self.learningrate = 0
   self.lrdecay = 0
   self.weightdecay = 0
   
   self.constantinputsize = false
   self.inputsize = {0, 0}
   self.jittering = {0, 0}
   
   self.batchcount = 0
   self.gradupperbound = 1
   
   self.nclasses = 0
   self.confusion = 0
   
end


function NeuralNet:setNetwork(net)
   self.network=net
end

function NeuralNet:setNumclasses(nclasses)
   self.nclasses=nclasses
   self.confusion=optim.ConfusionMatrix(nclasses)
end


function NeuralNet:setCriterion(criterion)
   self.criterion=criterion
end


function NeuralNet:setMeanoverset(meanoverset)
   self.meanoverset=meanoverset
end


function NeuralNet:setDatasetdir(datasetdir)
   self.datasetdir=datasetdir
end


function NeuralNet:setTrainsetRange(first, last)
   self.trainset={first, last}
end


function NeuralNet:setTestsetRange(first, last)
   self.testset={first, last}
end


function NeuralNet:setBatchsize(batchsize)
   self.batchsize=batchsize
end


function NeuralNet:setCheckpoint(checkpointdir, checkpointname)
   self.checkpointdir=checkpointdir
   self.checkpointname=checkpointname
end


function NeuralNet:shuffleTrainset()
   self.batchshuffle=torch.randperm(self.trainset[2] - self.trainset[1])
end


function NeuralNet:setMomentum(momentum)
   self.momentum=momentum
end


function NeuralNet:setLearningrate(learningrate)
   self.learningrate=learningrate
end


function NeuralNet:setLRdecay(lrdecay)
   self.lrdecay=lrdecay
end


function NeuralNet:setWeightdecay(weightdecay)
   self.weightdecay=weightdecay
end

function NeuralNet:setGradupperbound(gradupperbound)
   self.gradupperbound=gradupperbound
end


function NeuralNet:setInputsize(constantinputsize, inputsize, jittering)
   self.constantinputsize=constantinputsize
   self.inputsize = inputsize
   self.jittering = jittering
end


function NeuralNet:getBatch(batchidx)
   local batchfile=torch.load(paths.concat(self.datasetdir, 'batch'..batchidx..'.t7'))
   local batch=batchfile[1]:float()
   batch:add(-1, self.meanoverset)
   local target=batchfile[2]
   return batch, target
end


function NeuralNet:train(nepochs)
   -- init 
   if self.batchcount == 0 then 
      self.shuffleTrainset()
   end
end












   
