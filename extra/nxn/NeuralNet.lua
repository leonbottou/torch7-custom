require 'optim'
require 'image'

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
      self.network = nil              -- should be nxn.Sequential
      self.criterion = nil            -- should be nxn.Criterion
      
      self.meanoverset = nil          -- should be a torch.Tensor() of the same type as the network input
      self.datasetdir = nil           -- should be a '/path/to/dataset'
      self.trainset = nil             -- should be a {first, last}
      self.trainsetsize = nil         -- should be last - first + 1
      self.testset = nil              -- should be a {first, last}
      self.batchsize = nil            -- should be an integer
      
      self.checkpointdir = nil        -- should be a '/path/to/checkpoint'
      self.checkpointname = nil       -- should be a 'filename'
      
      self.batchshuffle = nil         -- save the torch.randperm (shuffling order of the batches)
      
      -- optional stuff
      self.momentum = 0               -- should be between 0 and 1
      self.learningrate = 0           -- optional, but if you want to train... well, you know.
      self.lrdecay = 0                -- will decay like : LR(t) = learningrate / ( 1 + lrdecay * number of batches seen by the net )
      self.weightdecay = 0            -- will put a L2-norm penalty on the weights
      
      self.inputtype = 'torch.FloatTensor' -- stick to this if you want to CUDA your net
      self.horizontalflip = false     -- should be true or false, will flip horizontally your input before feeding them to the net
      
      self.epochshuffle = false       -- should be true or false (shuffle the minibatch order at the beginning of each epoch)
      self.epochcount = 0             -- where the network is at
      self.batchcount = 0             -- where the network is at
      self.gradupperbound = nil       -- L2-norm constraint on the gradients : if a gradient violates the constraint, it will be projected on the L2 unit-ball
      
      self.nclasses = nil             -- number of classes of the net output
      self.confusion = nil            -- confusion matrix, useful for monitoring the training
      
      self.costvalues = {}             -- we want to store the values of the cost during training
      self.testcostvalues = {}         -- we want to store the values of the cost during test passes
      
      self.lasttraincall = {}
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
   self.criterion.sizeAverage=false
end


function NeuralNet:setMeanoverset(meanoverset)
   self.meanoverset=meanoverset
end


function NeuralNet:setDatasetdir(datasetdir)
   self.datasetdir=datasetdir
end


function NeuralNet:setTrainsetRange(first, last)
   self.trainset={first, last}
   self.trainsetsize=last-first+1
   
end


function NeuralNet:setTestsetRange(first, last)
   self.testset={first, last}
end

function NeuralNet:setInputType(tensortype)
   self.inputtype=tensortype
end


function NeuralNet:setBatchsize(batchsize)
   self.batchsize=batchsize
end


function NeuralNet:setCheckpoint(checkpointdir, checkpointname)
   self.checkpointdir=checkpointdir
   self.checkpointname=checkpointname
end


function NeuralNet:saveNet()
   torch.save(paths.concat(self.checkpointdir, self.checkpointname), self)
end

function NeuralNet:setEpochShuffle(epochshuffle)
   self.epochshuffle=epochshuffle
end

function NeuralNet:shuffleTrainset()
   self.batchshuffle=torch.randperm(self.trainsetsize)
end

function NeuralNet:getBatchNum(idx)
   return self.trainset[1]+self.batchshuffle[idx]-1
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


function NeuralNet:setHorizontalflip(horizontalflip)
   self.horizontalflip=horizontalflip
end



-- you can change these to load another kind of batches...

function NeuralNet:getBatch(batchidx)
   local batchfile=torch.load(paths.concat(self.datasetdir, 'batch'..batchidx..'.t7'))
   local batch=batchfile[1]:type(self.inputtype)
   if self.meanoverset then
      batch:add(-1, self.meanoverset)
   end
   local target=batchfile[2]
   return batch, target
end

function NeuralNet:getTestBatch(batchidx)
   local batchfile=torch.load(paths.concat(self.datasetdir, 'batch'..batchidx..'.t7'))
   local batch=batchfile[1]:type(self.inputtype)
   if self.meanoverset then
      batch:add(-1, self.meanoverset)
   end
   local target=batchfile[2]
   return batch, target
end

--


function NeuralNet:getNumBatchesSeen()
   return self.epochcount*self.trainsetsize+self.batchcount
end


function nxn.NeuralNet:showL1Filters()
   local p,g = self.network:parameters()
   foo=p[1]:float()
   foo=foo:transpose(3,4):transpose(1,2):transpose(2,3)
   image.display({image=foo, zoom=3, padding=1}) 
end




function nxn.NeuralNet:plotError()
   require 'gnuplot'
   local npoints=#self.costvalues
   local costvector=torch.Tensor(npoints)
   for i=1,npoints do
      costvector[{i}]=self.costvalues[i][3]
   end
   
   local ntestpoints=#self.testcostvalues
   local testcostvector=torch.Tensor(ntestpoints)
   local testcostindices=torch.Tensor(ntestpoints)
   
   for i=1,ntestpoints do
      testcostvector[{i}]=self.testcostvalues[i][2]
      testcostindices[{i}]=self.testcostvalues[i][1]
   end
   
   gnuplot.plot({torch.range(1,npoints)/self.trainsetsize, costvector, '-'},{'Train set cost', torch.range(1,npoints)/self.trainsetsize, costvector, '-'},{'Validation set cost', testcostindices/self.trainsetsize, testcostvector,'-'})
   
end


function NeuralNet:resume()
   self:train(self.lasttraincall[1],self.lasttraincall[2],self.lasttraincall[3])
end


function NeuralNet:train(nepochs, savefrequency, measurementsfrequency)
   self.lasttraincall={nepochs, savefrequency, measurementsfrequency}
   -- do a lot of tests and return errors if necessary :
   if not nepochs then
      error('NeuralNet:train(n [, fsave, fmeas]), will train until epoch n is reached (starts at 0), save every fsave batches, take measurements every fmeas batches (you can set these to nil)') 
   end
   
   if not self.network then
      error('no network : use NeuralNet:setNetwork(net)') 
   end
   
   if not self.criterion then
      error('no criterion : use NeuralNet:setCriterion(criterion)') 
   end
   
   if not self.datasetdir then
      print('no dataset folder : use NeuralNet:setDatasetdir("/path/to/dataset"), or write your own NeuralNet:getBatch(idx) function') 
   end
   
   if not self.trainset then
      error('no training set range : use NeuralNet:setTrainsetRange(first, last)') 
   end
   
   if not self.batchsize then
      error('no batch size set : use NeuralNet:setBatchsize(first, last)') 
   end
   
   if measurementsfrequency and (not self.testset) then
      error('no validation set range : use NeuralNet:setTestsetRange(first, last)') 
   end
   
   if savefrequency and ((not self.checkpointdir) or (not self.checkpointname)) then
      error('no checkpoint : use NeuralNet:setCheckpoint("/path/to/checkpoint", "checkpointname")')
   end
   
   if not self.nclasses then
      error('no information on the number of classes : use NeuralNet:setNumclasses(n)') 
   end
   
   
   
   -- put all modules in train mode (useful for dropout)
   self.network:setTestMode(false)
   
   
   
   
   time=torch.Timer()
   -- training loop
   while self.epochcount<nepochs do
      -- init 
      if self.batchcount > self.trainsetsize then
         self.epochcount = self.epochcount + 1 
            self.batchcount = 0
      end   
      
      if self.batchcount == 0 then 
         if self.epochshuffle or self.epochcount==0 then
            self:shuffleTrainset()
         end
         self.batchcount = 1
      end
      
      -- get proper batch
      local batchidx = self:getBatchNum(self.batchcount)
      self.batchcount = self.batchcount + 1
      
      local input, target = self:getBatch(batchidx)
      
      
      -- forward 
      self.network:forward(input)
      self.criterion:forward(self.network.output, target)
      
      -- confusion : only interesting for classification
      if self.network.output:dim()==2 then
         for k=1,self.batchsize do
            self.confusion:add(self.network.output[{k,{}}], target[{k}])
         end
         self.confusion:updateValids()
      end
      
      print('epoch : '..self.epochcount..', batch num : '..(self.batchcount-1)..' idx : '..batchidx..', cost : '..self.criterion.output/self.batchsize..', average valid % : '..(self.confusion.averageValid*100)..', time : '..time:time().real)   
         table.insert(self.costvalues, {self:getNumBatchesSeen()-1, batchidx, self.criterion.output/self.batchsize, self.confusion.averageValid*100})
      self.confusion:zero()
      time:reset()
      
      
      
      
      -- backward :
      
      local df_do=self.criterion:backward(self.network.output, target)
      local currentlr = self.learningrate / (1 + self.lrdecay * self:getNumBatchesSeen())
      local params, gradients =self.network:parameters()
      
      -- apply momentum :
      for idx=1,#gradients do 
         gradients[idx]:mul(self.momentum)
      end
      
      -- compute and accumulate gradients
      self.network:backward(input, df_do, currentlr/self.batchsize)
      
      -- apply weight decay :
      for idx=1,#gradients do
         gradients[idx]:add(self.weightdecay*currentlr, params[idx])
      end
      
      -- clip gradients
      if self.gradupperbound then
         for idx=1,#gradients do
            local gnorm=gradients[idx]:norm()
            if gnorm > self.gradupperbound then
               gradients[idx]:mul(self.gradupperbound/gnorm)
            end
         end
      end
      self.network:updateParameters(1)
      
      if measurementsfrequency then
         if math.mod(self:getNumBatchesSeen(),measurementsfrequency)==0 then
            self:showL1Filters()
            self:plotError()
            
            for idx=1,#params do 
               --print('param id : '.. idx)
               local WorB
               if math.mod(idx,2)==1 then WorB=' weight' else WorB=' bias' end
               print('module '..math.ceil(idx/2)..WorB..' mean : '..(params[idx]:mean())..', grad/LR mean : '..(gradients[idx]:mean()*currentlr))
               print('module '..math.ceil(idx/2)..WorB..' std  : '..(params[idx]:std())..', grad/LR std  : '..(gradients[idx]:std()*currentlr))
               print(' ')
            end
            
            local meancost=0
            -- run on validation set :
            self.network:setTestMode(true)
            for valbatchidx=self.testset[1],self.testset[2] do
               local valbatch,valtarget=self:getTestBatch(valbatchidx)  
               
               self.network:forward(valbatch)
               self.criterion:forward(self.network.output, valtarget)
               meancost=meancost+self.criterion.output
               if self.network.output:dim()==2 then
                  for k=1,batchsize do
                     self.confusion:add(self.network.output[{k,{}}], valtarget[{k}])
                  end
               end
            end
            self.network:setTestMode(false)
            meancost=meancost/(self.testset[2]-self.testset[1]+1)/self.batchsize
            self.confusion:updateValids()
            print('mean cost on validation set : '..meancost.. ', average valid % : '..(self.confusion.averageValid*100))
            table.insert(self.testcostvalues, {self:getNumBatchesSeen(), meancost, self.confusion.averageValid*100})
            self.confusion:zero()
            
         end
      end
      
      if savefrequency then
         if math.mod(self:getNumBatchesSeen()-1,savefrequency)==0 then
            self:saveNet()
         end
      end
      
      
   end
end




















