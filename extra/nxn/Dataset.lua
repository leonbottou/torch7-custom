local Dataset = torch.class('nxn.Dataset')

-- Dataset:generateSample(idx) should generate one sample of fixed size and return sample,target
-- If you want to change stuff for your own dataset, this is the only function you want to touch

-- Dataset:generate() should take care of the rest, and will store a batch of data as a ByteTensor, and targets as a FloatTensor
-- Dataset:generate() can stop&resume, as it saves its state in targetDir after each batch
-- to resume, Dataset:setTargetDir('path/to/target'), then Dataset:resume()

-- note : if you want to use the example generator you have to build your dataset as a table of samples : 
-- dataTable={sample1, sample2, sample3} where sample1={'path/to/image.jpg', label}

function Dataset:__init()
   self.dataTable={}
   self.targetDir=nil
   self.batchSize=128
   self.shuffle=nil
   self.meanoverset=nil
   self.nextbatch=0
   self.x=256
   self.y=256
end

function Dataset:setDataTable(dataTable)
   self.dataTable=dataTable
end

function Dataset:setSizes(x,y)
   self.x=x
   self.y=y
end

function Dataset:setTargetDir(targetDir)
   self.targetDir=targetDir
end

function Dataset:setBatchSize(batchSize)
   self.batchSize=batchSize
end

function Dataset:generateSample(idx)
   require 'image'
   -- this example generator will load an image from the data table and return a 256*256 RGB image
   -- if the images are already in 256*256 RGB format, then they are just returned with the correct transposition (y,x,channels)
   local foo=self.dataTable[idx]
   local img0=image.load(foo[1], 3, 'byte')
   
   local sample
   if (#img0)[1]~=3 or (#img0)[2]~=self.y or (#img0)[3]~=self.x then
      sample=image.scale(img0, 256, 256):transpose(1,2):transpose(2,3)
   else
      sample=img0:transpose(1,2):transpose(2,3)
   end
   
   return sample, torch.FloatTensor(1):fill(foo[2])
end

function Dataset:resume()
   self=torch.load(paths.concat(self.targetDir, 'dataGeneratorState.t7'))
   self:generateSet()
end

function Dataset:generateSet(shuffleorder)
   shuffleorder=shuffleorder or 1
   -- is it initialized ? if no, shuffle, and set self.nextbatch to 1
   
   local sampleExample,targetExample = self:generateSample(1)
   
   
   if self.nextbatch==0 then
      if shuffleorder==1 then 
         self.shuffle=torch.randperm(#self.dataTable)
      else
         self.shuffle=torch.range(1,#self.dataTable)
      end
      self.meanoverset=torch.FloatTensor(#sampleExample):fill(0)
      self.nextbatch=1
   end

   local numbatches=math.floor(#self.dataTable/self.batchSize)
   
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


   for batchidx=self.nextbatch,numbatches do 
      local sampleBatch=torch.ByteTensor(batchSampleDims)
      local targetBatch=torch.FloatTensor(batchTargetDims)
      
      for imgidx=1,self.batchSize do
         local currentidx=self.shuffle[(batchidx-1)*self.batchSize+imgidx]
         local sample, target = self:generateSample(currentidx)
         sampleBatch:select(1,imgidx):copy(sample)
         targetBatch:select(1,imgidx):copy(target)
      end
      
      batchfile={sampleBatch, targetBatch}
      torch.save(paths.concat(self.targetDir, 'batch'..batchidx..'.t7'), batchfile)
      
      self.meanoverset:add(sampleBatch:float():mean(1):select(1,1))
      self.nextbatch=self.nextbatch+1

      torch.save(paths.concat(self.targetDir, 'dataGeneratorState.t7'), self)
      print('Batch '..batchidx..' / '..numbatches..' : done.')

      collectgarbage()
   end

   self.meanoverset:div(numbatches)
   torch.save(paths.concat(self.targetDir, 'meanoverset.t7'), self.meanoverset)
end

