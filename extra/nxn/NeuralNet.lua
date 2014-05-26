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
      self.trainset = nil             -- should be a {first, last}
      self.trainsetsize = nil         -- should be last - first + 1
      self.testset = nil              -- should be a {first, last}
      
      self.checkpointdir = nil        -- should be a '/path/to/checkpoint'
      self.checkpointname = nil       -- should be a 'filename'
      self.vizdir = nil               -- should be a '/path/to/vizs'
      
      self.batchshuffle = nil         -- save the torch.randperm (shuffling order of the batches)
      
      self.epochshuffle = false       -- should be true or false (shuffle the minibatch order at the beginning of each epoch)
      self.epochcount = 0             -- where the network is at
      self.batchcount = 0             -- where the network is at

      self.nclasses = nil             -- number of classes of the net output
      self.confusion = nil            -- confusion matrix, useful for monitoring the training
      
      self.costvalues = {}             -- we want to store the values of the cost during training
      self.testcostvalues = {}         -- we want to store the values of the cost during test passes
      
      self.lasttraincall = {}
      self.gpumode=false
end

local function zapTensor(a)
   if a then 
      a:resize(0)
      a:storage():resize(0) 
   end
end

function NeuralNet:setNetwork(net)
   self.rawnetwork=net
   self:GPUWrap()
end

function NeuralNet:GPUWrap()
   if cutorch and self.rawnetwork:isGPUCompatible() then
      self.rawnetwork:cuda()
      self.network=nxn.Sequential()
      self.network:add(nxn.Copy('torch.FloatTensor', 'torch.CudaTensor'))
      self.network:add(self.rawnetwork)
      self.network:add(nxn.Copy('torch.CudaTensor', 'torch.FloatTensor'))
      self.gpumode=true
   else
      self.network=self.rawnetwork
      self.gpumode=false
   end
   return self.network
end

function NeuralNet:cleanNetwork()
   self.network:clean()
end

function NeuralNet:setNumclasses(nclasses)
   self.nclasses=nclasses
   self.confusion=optim.ConfusionMatrix(nclasses)
end

function NeuralNet:setCriterion(criterion)
   self.criterion=criterion
   self.criterion.sizeAverage=false
end

function NeuralNet:setDataset(dataset)
   -- we want a nxn.Dataset here
   self.dataset=dataset
end

function NeuralNet:setTrainsetRange(first, last)
   local numbatches = self.dataset:getNumBatches()
   if first >= 1 and last >= first and last <= numbatches then
      self.trainset={first, last}
      self.trainsetsize=last-first+1
   else error(' ... ')
   end
end

function NeuralNet:setTestsetRange(first, last)
   local numbatches = self.dataset:getNumBatches()
   if first >= 1 and last >= first and last <= numbatches then
      self.testset={first, last}
   else error(' ... ')
   end
end

function NeuralNet:setCheckpoint(checkpointdir, checkpointname)
   self.checkpointdir=checkpointdir
   self.checkpointname=checkpointname
end

function NeuralNet:saveNet()
   self:cleanNetwork()
   if self.gpumode then 
      self.rawnetwork:float()
      torch.save(paths.concat(self.checkpointdir, self.checkpointname), self)
      self.rawnetwork:cuda()
   else
      torch.save(paths.concat(self.checkpointdir, self.checkpointname), self)
   end   
end

function NeuralNet:setVisualizationDir(vizdir)
   self.vizdir=vizdir
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

function NeuralNet:getByName(name)
   return self.network:getByName(name)
end

function NeuralNet:__call__(name)
   return self.network:getByName(name)
end

function NeuralNet:setSaveMem(bool)
   return self.network:setSaveMem(bool)
end

-- you can change these to load another kind of batches...

function NeuralNet:getBatch(batchidx)
   return self.dataset:getBatch(batchidx)
end

function NeuralNet:cacheBatch(batchidx)
   return self.dataset:cacheBatch(batchidx)
end

function NeuralNet:getTestBatch(batchidx)
   return self:getBatch(batchidx)
end

--


function NeuralNet:getNumBatchesSeen()
   return self.epochcount*self.trainsetsize+self.batchcount
end


function NeuralNet:showL1Filters()
   local p,g = self.network:parameters()
   local foo=p[1]:float()
   foo=foo:transpose(3,4):transpose(2,3)
   if self.vizdir then
      img = image.toDisplayTensor({input=foo, padding=1, zoom=3})
      image.savePNG(paths.concat(self.vizdir, 'l1filters.png'), img)
   else
      image.display({image=foo, zoom=3, padding=1}) 
   end
end

function NeuralNet:updateConfusion(target)
   if self.confusion then
      if target:size(1) ~= self.network.output:size(1) then error('network output and target sizes are inconsistent') end
      if self.network.output:dim()==2 then
         for k=1,target:size(1) do
            self.confusion:add(self.network.output[{k,{}}], target[{k}])
         end
      end
   end
end


function NeuralNet:plotError()
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
   
   if self.vizdir then
      local fignum = gnuplot.pngfigure(paths.concat(self.vizdir, 'error.png'))
   end
   
   if ntestpoints>0 then
   gnuplot.plot({torch.range(1,npoints)/self.trainsetsize, costvector, '-'},
   {'Train set cost', torch.range(1,npoints)/self.trainsetsize, costvector, '-'},
   {'Validation set cost', testcostindices/self.trainsetsize, testcostvector,'-'})
   else
      gnuplot.plot({torch.range(1,npoints)/self.trainsetsize, costvector, '-'},
      {'Train set cost', torch.range(1,npoints)/self.trainsetsize, costvector, '-'})
   end
   
   if self.vizdir then
      gnuplot.close(fignum);
   end
end

function NeuralNet:setTestMode(value)
   self.network:setTestMode(value)
end


function NeuralNet:resume()
   self:train(self.lasttraincall[1],self.lasttraincall[2],self.lasttraincall[3])
end

function NeuralNet:testBatch(valbatchidx, mod)
  local valbatch, valtarget=self:getTestBatch(valbatchidx)
  mod = mod or self.network

  self:setTestMode(true)

  self.network:forward(valbatch)
  self.criterion:forward(self.network.output, valtarget)
  
  if self.confusion then self:updateConfusion(valtarget)  end

  self:setTestMode(false)
  return self.criterion.output, valbatch:size(1), valtarget, mod.output
end

function NeuralNet:test()
   local params, gradients =self.network:parameters()
   local meancost=0
   local numexamples=0
   -- run on validation set :
   for valbatchidx=self.testset[1],self.testset[2] do
    crit_out, batch_numexamples, _, _ = self:testBatch(valbatchidx)
    meancost = meancost + crit_out
    numexamples = numexamples + batch_numexamples
   end
   meancost=meancost/numexamples

   local avgvalid = -1
   if self.confusion then 
      self.confusion:updateValids() 
      avgvalid = self.confusion.averageValid*100
      self.confusion:zero()
      print('mean cost on validation set : '..meancost.. ', average valid % : '..avgvalid)
   else
      print('mean cost on validation set : '..meancost)
   end
   table.insert(self.testcostvalues, {self:getNumBatchesSeen(), meancost, avgvalid})
   
end


function NeuralNet:measure()
   local params, gradients = self.network:parameters()
   if self.epochcount==0 then self:showL1Filters() end
   
   for idx=1,#params do 
      --print('param id : '.. idx)
      local WorB
      if math.mod(idx,2)==1 then WorB=' weight' else WorB=' bias' end
      print('module '..math.ceil(idx/2)..WorB..' mean : '..(params[idx]:mean())..', grad mean : '..(gradients[idx]:mean()))
      print('module '..math.ceil(idx/2)..WorB..' std  : '..(params[idx]:std())..', grad std  : '..(gradients[idx]:std()))
      print(' ')
   end
end


function NeuralNet:forwardprop(input, target, timer, batchidx)
   self.network:forward(input)
   self.criterion:forward(self.network.output, target)
   
   -- confusion : only interesting for classification
   if self.confusion then 
      self:updateConfusion(target)
      self.confusion:updateValids()
   print('epoch : '..self.epochcount..', batch num : '..(self.batchcount-1)..' idx : '..batchidx..', cost : '..self.criterion.output/input:size(1)..', average valid % : '..(self.confusion.averageValid*100)..', time : '..time:time().real)   
      table.insert(self.costvalues, {self:getNumBatchesSeen()-1, batchidx, self.criterion.output/input:size(1), self.confusion.averageValid*100})
      self.confusion:zero()
   else
   print('epoch : '..self.epochcount..', batch num : '..(self.batchcount-1)..' idx : '..batchidx..', cost : '..self.criterion.output/input:size(1)..', time : '..time:time().real)   
      table.insert(self.costvalues, {self:getNumBatchesSeen()-1, batchidx, self.criterion.output/input:size(1), -1})
   end   
end


function NeuralNet:backpropUpdate(input, df_do, target, lr)
   local params, gradients =self.network:parameters()
   
   -- compute and accumulate gradients
   self.network:backward(input, df_do, lr/input:size(1))

   self.network:updateParameters()
   
   self.batchcount = self.batchcount + 1
end



function NeuralNet:train(nepochs, savefrequency, measurementsfrequency)
   self.lasttraincall={nepochs, savefrequency, measurementsfrequency}
   self:GPUWrap()
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
   
   if not self.trainset then
      error('no training set range : use NeuralNet:setTrainsetRange(first, last)') 
   end

   if measurementsfrequency and (not self.testset) then
      error('no validation set range : use NeuralNet:setTestsetRange(first, last)') 
   end
   
   if savefrequency and ((not self.checkpointdir) or (not self.checkpointname)) then
      error('no checkpoint : use NeuralNet:setCheckpoint("/path/to/checkpoint", "checkpointname")')
   end
   
   if not self.nclasses then
      print('no information on the number of classes : use NeuralNet:setNumclasses(n)') 
   end
  
   time=torch.Timer()
   -- training loop
   while self.epochcount<nepochs do
      -- put all modules in train mode (useful for dropout)
      self:setTestMode(false)
      self.network:setBackProp()   

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
      if self.batchcount<self.trainsetsize then
         local nextbatchidx = self:getBatchNum(self.batchcount+1)
         self:cacheBatch(nextbatchidx)
      end

      local input, target = self:getBatch(batchidx)
      
      -- forward 
      local successf, errormsgf = pcall (self.forwardprop, self, input, target, time, batchidx)
      if not successf then 
         if errormsgf=='stop' then
            print('stopped during forward prop')
            return
         else
            error(errormsgf..' during forward prop') 
         end
      end

      time:reset()
      
      -- backward :
      local df_do=self.criterion:backward(self.network.output, target)
      local currentlr = 1
      local successb, errormsgb = pcall(self.backpropUpdate, self, input, df_do, target, currentlr)
      if not successb then 
         if errormsgb=='stop' then
            print('stopped during backprop')
            return
         else
            error(errormsgb..' during backprop') 
         end
      end
      
      if measurementsfrequency then
         if math.mod(self:getNumBatchesSeen(),measurementsfrequency)==0 then
            self:measure()
            self:test()
            self:plotError()
         end     
      end
      
      if savefrequency then
         if math.mod(self:getNumBatchesSeen(),savefrequency)==0 then
            self:saveNet()
         end
      end
      
      
   end
end




















