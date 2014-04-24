local Dataset = torch.class('nxn.Dataset')

-- Dataset:generateSample(idx) should generate one sample of fixed size and return sample,target
-- If you want to change stuff for your own dataset, this is the only function you want to modify

-- Dataset:generateSet() should take care of the rest, and will store a batch of data as a ByteTensor, and targets as a FloatTensor
-- Dataset:generateSet() can stop&resume, as it saves its state in targetDir after each batch

-- easy mode : (all images will be resized (warped) to width*height)
-- 1) put all images of a same class in a folder
-- 2) put all these folders in a folder "path/to/data"
-- 3) foo=nxn.Dataset()
-- 4) foo:automatic('path/to/data', 'path/to/output', width, height)
-- if your computer crashes : run nxn.Dataset.resume('path/to/output')


-- advanced mode :
-- 1) foo=nxn.Dataset()
-- 2) foo:setBatchSize(bs)
-- 3) foo:setSize(numSamples)
-- 4) foo:shuffleOrder() if you want your dataset shuffled (which may not be the case)
-- 5) define foo:generateSample(idx)
--    => must output idx-th sample, of fixed size across the dataset 
--    => function should finish with "return sample, target"
--    => if intermediate data (e.g. list of files) must be used, put it in the nxn.Dataset object 
-- (it is the one that is saved during generation for resuming)
-- 6) foo:setTargetDir('path/to/output')
-- 7) foo:generateSet()
-- if your computer crashes : run nxn.Dataset.resume('path/to/output')


-- usage : 
-- NeuralNet:setDataset(nxn.Dataset.loadSet('path/to/output'))


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
   return math.ceil(self.numSamples/self.batchSize)
end

function Dataset:setSize(numSamples)
   self.numSamples=numSamples
   self.shuffle=torch.range(1,self.numSamples)
   print('shuffle the order of samples with Dataset:shuffleOrder()')
end

function Dataset:shuffleOrder()
   self.shuffle=torch.randperm(self.numSamples)
   print('dataset shuffled')
end

function Dataset:setTargetDir(targetDir)
   if paths.filep(paths.concat(targetDir, 'dataGeneratorState.t7')) then
      error('dataset already exists in '..targetDir..' , please use ds=nxn.Dataset.loadSet("/path/to/target/folder/")')
   end
   self.targetDir=targetDir
end

function Dataset:setBatchSize(batchSize)
   self.batchSize=batchSize
end

function Dataset:generateSample(idx)
   require 'image'
   local path, target -- = ????
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
   local ds=torch.load(paths.concat(folder, 'dataset.t7'))
   ds.targetDir=folder
   return ds
end

function Dataset.resume(folder)
   local ds=torch.load(paths.concat(folder, 'dataGeneratorState.t7'))
   ds:generateSet()
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
   
   local batchfill=0

   for imgidx=1,self.batchSize do
      if (batchidx-1)*self.batchSize+imgidx <= self.shuffle:size(1) then
         local currentidx=self.shuffle[(batchidx-1)*self.batchSize+imgidx]
         local sample, target = self:generateSample(currentidx)
         sampleBatch:select(1,imgidx):copy(sample)
         targetBatch:select(1,imgidx):copy(target)
         batchfill=imgidx
      end
   end
   sampleBatch=sampleBatch:narrow(1, 1, batchfill)
   targetBatch=targetBatch:narrow(1, 1, batchfill)
  
   --print('Batch '..batchidx..' : generated.')
   if targetBatch:numel()==targetBatch:size(1) then
      local u=targetBatch
      targetBatch=torch.FloatTensor(batchfill):copy(u)
   end
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
   
   local dataset=nxn.Dataset()
   local ls=torch.LongStorage(self.meanoverset:dim()+1):fill(1)
   dataset.realMeanOverSet=torch.repeatTensor(self.meanoverset, ls)
   dataset.batchSize=self.batchSize
   dataset.classlist=self.classlist
   dataset.numSamples=self.numSamples
   dataset.getBatch=self.getBatch
   dataset.cacheBatch=self.cacheBatch
   dataset.getTestBatch=self.getTestBatch
   torch.save(paths.concat(self.targetDir, 'dataset.t7'), dataset)
   
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

require 'torchffi'
function Dataset:automatic(source, target, xsize, ysize)
   if not self.bigStringTensor then
      -- this will generate a dataset from whatever is in the source folder
      -- one class per subfolder
      -- only classification
      dirlist=paths.dir(source)
      table.sort(dirlist)
      classlist={}
      classcount=0
      for a, b in ipairs(dirlist) do
         if paths.dirp(paths.concat(source, b)) and b~='.' and b~='..' then
            classlist[b]= classcount+1
            classcount=classcount+1
         end
      end
      
      -- find number of samples
      numSamples=0
      
      for class, idx in pairs(classlist) do
         classdir=paths.concat(source, class)
         print(classdir)
         numSamples=numSamples+#paths.dir(classdir)-2
      end
      
      -- arbitrary upper bound
      maxStringLength = 200 
         
         -- allocate CharTensor
         bigStringTensor = torch.CharTensor(numSamples, maxStringLength)
      bst_data=torch.data(bigStringTensor)
      labelTensor = torch.LongTensor(numSamples)
      
      bstcount=0
      for class, idx in pairs(classlist) do
         classdir=paths.concat(source, class)
         filenames=paths.dir(classdir)
         for a,b in ipairs(paths.dir(classdir)) do
            if paths.filep(paths.concat(classdir,b)) then
               ffi.copy(bst_data + (bstcount*maxStringLength), paths.concat(classdir,b))
               bstcount=bstcount+1
               labelTensor[{bstcount}]=idx
            end
         end
      end

   
   function self.readStringInTensor(bst, idx)
      if idx<=bst:size(1) then
         local bst_data=torch.data(bst)
         return ffi.string(bst_data + ((idx-1) * bst:size(2)))
      end
   end
   
   function self:generateSample(idx)
      require 'image'
      local path=self.readStringInTensor(self.bigStringTensor, idx)
      local img0=image.load(path, 3, 'byte')
      local cls=self.labelTensor[{idx}]
      
      local sample
      if (#img0)[1]~=3 or (#img0)[2]~=ysize or (#img0)[3]~=xsize then
         sample=image.scale(img0, xsize, ysize):transpose(1,2):transpose(2,3)
      else
         sample=img0:transpose(1,2):transpose(2,3)
      end
      
      return sample, torch.FloatTensor(1):fill(cls)
   end 
      
   self:setSize(numSamples)
   self:shuffleOrder()
   self:setTargetDir(target)
      
   self.labelTensor = labelTensor
   self.classlist = classlist
   self.bigStringTensor = bigStringTensor
   
   end
   
   self:generateSet()

end







