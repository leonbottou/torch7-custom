local Dataset = torch.class('nxn.Dataset')

-- Dataset:generateSample(idx) should generate one sample of fixed size and return sample,target
-- If you want to change stuff for your own dataset, this is the only function you want to touch

-- Dataset:generate() should take care of the rest, and will store a batch of data as a ByteTensor, and targets as a FloatTensor
-- Dataset:generate() can stop&resume, as it saves its state in targetDir after each batch
-- to resume, Dataset:setTargetDir('path/to/target'), then Dataset:resume()

-- note : if you want to use the example generator you have to build your dataset as a table of samples : 
-- dataTable={sample1, sample2, sample3} where sample1={'path/to/image.jpg', label}

-- if you want your dataset shuffled, do Dataset:shuffleOrder() before generating stuff...

function Dataset:__init(folder)

   self.batchSize=128
   self.shuffle=nil
   self.meanoverset=nil
   self.nextbatch=0
   self.numSamples=nil
   self.finished=false
   self.realMeanOverSet=nil

   if folder then self:setTargetDir(folder) end
   
end

function Dataset:getNumBatches()
   return math.floor(self.numSamples/self.batchSize)
end

function Dataset:setSize(numSamples)
   self.numSamples=numSamples
   self.shuffle=torch.range(1,self.numSamples)
   print('shuffle the order of samples with Dataset:shuffleOrder()')
end

function Dataset:shuffleOrder()
   self.shuffle=torch.randperm(self.numSamples)
end

function Dataset:setTargetDir(targetDir)
   self.targetDir=targetDir
   if paths.filep(paths.concat(self.targetDir, 'dataGeneratorState.t7')) then
      self:load()
      print ('loaded dataset from '..self.targetDir)
   end
end

function Dataset:setBatchSize(batchSize)
   self.batchSize=batchSize
end

   -- this function has to return a path to an image and a label
function Dataset:getInstance(idx)
   return path, labelnumber
end

function Dataset:generateSample(idx)
   require 'image'
   -- this example generator will load an RGB image from the data table and return a 256*256 RGB image + the label
   -- if the images are already in self.x*self.y RGB format, then they are just returned with the correct transposition (y,x,channels)

   local path, target = self:getInstance(idx)
   local img0=image.load(path, 3, 'byte')
   
   local sample
   if (#img0)[1]~=3 or (#img0)[2]~=256 or (#img0)[3]~=256 then
      sample=image.scale(img0, 256, 256):transpose(1,2):transpose(2,3)
   else
      sample=img0:transpose(1,2):transpose(2,3)
   end
   
   return sample, torch.FloatTensor(1):fill(target)
end

function Dataset.loadSet(folder)
   return torch.load(paths.concat(folder, 'dataGeneratorState.t7'))
end

function Dataset:load()
   local foo=torch.load(paths.concat(self.targetDir, 'dataGeneratorState.t7'))
   self.getInstance=foo.getInstance
   self.generateSample=foo.generateSample
   self.numSamples=foo.numSamples
   self.targetDir=foo.targetDir
   self.batchSize=foo.batchSize
   self.shuffle=foo.shuffle
   self.meanoverset=foo.meanoverset
   self.nextbatch=foo.nextbatch
   self.finished=foo.finished
end

function Dataset:resume()
   self:generateSet()
end


function Dataset:generateBatch(batchidx)
   local sampleExample,targetExample = self:generateSample(1)
   local numbatches=self:getNumBatches()

   if batchidx > numbatches then 
      error('idx must be < numbatches (= '..numbatches..')')
   end
   
   local sampleExampleDims=#(#sampleExample)
   local targetExampleDims=#(#targetExample)

   local batchSampleDims=torch.LongStorage(1+sampleExampleDims)
   local batchTargetDims=torch.LongStorage(1+targetExampleDims)
   
   batchSampleDims[1]=self.batchSize
   for d=1,sampleExampleDims do
      batchSampleDims[1+d]=(#sampleExample)[d]
   end
   
   batchTargetDims[1]=self.batchSize
   for d=1,targetExampleDims do
      batchTargetDims[1+d]=(#targetExample)[d]
   end

   local sampleBatch=torch.ByteTensor(batchSampleDims)
   local targetBatch=torch.FloatTensor(batchTargetDims)
   
   for imgidx=1,self.batchSize do
      local currentidx=self.shuffle[(batchidx-1)*self.batchSize+imgidx]
      local sample, target = self:generateSample(currentidx)
      sampleBatch:select(1,imgidx):copy(sample)
      targetBatch:select(1,imgidx):copy(target)
   end
   
   --print('Batch '..batchidx..' : generated.')
   
   collectgarbage()   
   return sampleBatch, targetBatch

end



function Dataset:generateSet()
   local sampleExample,targetExample = self:generateSample(1)
   
   if self.nextbatch==0 then
      self.meanoverset=torch.FloatTensor(#sampleExample):fill(0)
      self.nextbatch=1
   end

   local numbatches=self:getNumBatches()

   for batchidx=self.nextbatch,numbatches do 
      
      local sampleBatch, targetBatch
      sampleBatch, targetBatch = self:generateBatch(batchidx)
      local batchfile={sampleBatch, targetBatch}
      torch.save(paths.concat(self.targetDir, 'batch'..batchidx..'.t7'), batchfile)
      
      self.meanoverset:add(sampleBatch:float():mean(1):select(1,1))
      self.nextbatch=self.nextbatch+1

      torch.save(paths.concat(self.targetDir, 'dataGeneratorState.t7'), self)
      print('Batch '..batchidx..' / '..numbatches..' : done.')

      collectgarbage()
   end

   self.meanoverset:div(numbatches)
   torch.save(paths.concat(self.targetDir, 'meanoverset.t7'), self.meanoverset)
   self.finished=true
   torch.save(paths.concat(self.targetDir, 'dataGeneratorState.t7'), self)
end


function Dataset:expandMeanoverset(batch)
   if not self.realMeanOverSet then
      local meanoverset = torch.load(paths.concat(self.targetDir, 'meanoverset.t7'))
      local ls=torch.LongStorage(meanoverset:dim()+1):fill(1)
      self.realMeanOverSet=torch.repeatTensor(meanoverset, ls)
   end
   if not self.realMeanOverSetBatch then
      self.realMeanOverSetBatch=self.realMeanOverSet:expandAs(batch):contiguous()
   end
   if (self.realMeanOverSetBatch:size(1) ~= batch:size(1)) then
      self.realMeanOverSetBatch=self.realMeanOverSet:expandAs(batch):contiguous()
   end
end

function Dataset:getBatch(batchidx)
   collectgarbage()
   local batchfile=torch.load(paths.concat(self.targetDir, 'batch'..batchidx..'.t7'))
   local batch=batchfile[1]:float()
   self:expandMeanoverset(batch)
   if self.realMeanOverSetBatch then
      batch:add(-1, self.realMeanOverSetBatch)
   end
   local target=batchfile[2]
   return batch, target
end

function Dataset:cacheBatch(batchidx)
   os.execute('cp '..paths.concat(self.targetDir, 'batch'..batchidx..'.t7')..' /dev/null & ')
end

function Dataset:getTestBatch(batchidx)
   local batch, target = self:getBatch(batchidx)
   return batch, target
end

--

