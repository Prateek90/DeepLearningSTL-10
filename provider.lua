require 'nn'
require 'image'
require 'xlua'
require 'cutorch'
dofile 'patch_extract.lua'
dofile 'kmeans.lua'

torch.setdefaulttensortype('torch.FloatTensor')
--unlabel_batchSize=params.unlabelbatchsize
--unlabelsize=100000

-- parse STL-10 data from table into Tensor
function parseDataLabel(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
    t[idx]:copy(this_d[j])
    l[idx] = i
    idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t, l
end

function parseUnlabelData(d, numSamples, numChannels, height, width)
   local t = torch.ByteTensor(numSamples, numChannels, height, width)
   --local l = torch.ByteTensor(numSamples)
   local idx = 1
   for i = 1, #d do
      local this_d = d[i]
      for j = 1, #this_d do
    t[idx]:copy(this_d[j])
    --l[idx] = i
    idx = idx + 1
      end
   end
   assert(idx == numSamples+1)
   return t
end

local Provider = torch.class 'Provider'

function Provider:__init(full)
  local trsize = 4000
  local valsize = 1000  -- Use the validation here as the valing se
  unlabelsize=100000
  local testsize=8000
  local channel = 3
  local height = 96
  local width = 96

  -- download dataset
  if not paths.dirp('stl-10') then
     os.execute('mkdir stl-10')
     local www = {
         train = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/train.t7b',
         val = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/val.t7b',
         extra = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/extra.t7b',
         test = 'https://s3.amazonaws.com/dsga1008-spring16/data/a2/test.t7b'
     }

     os.execute('wget ' .. www.train .. '; '.. 'mv train.t7b stl-10/train.t7b')
     os.execute('wget ' .. www.val .. '; '.. 'mv val.t7b stl-10/val.t7b')
     os.execute('wget ' .. www.test .. '; '.. 'mv test.t7b stl-10/test.t7b')
     os.execute('wget ' .. www.extra .. '; '.. 'mv extra.t7b stl-10/extra.t7b')
  end

  local raw_train = torch.load('stl-10/train.t7b')
  local raw_val = torch.load('stl-10/val.t7b')
  local raw_unlabel = torch.load('stl-10/extra.t7b')
  local raw_test = torch.load('stl-10/test.t7b')

  -- load and parse dataset
  self.trainData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return trsize end
  }
  self.trainData.data, self.trainData.labels = parseDataLabel(raw_train.data,
                                                   trsize, channel, height, width)
  local trainData = self.trainData
  self.valData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return valsize end
  }
  self.valData.data, self.valData.labels = parseDataLabel(raw_val.data,
                                                 valsize, channel, height, width)
  local valData = self.valData
  
    self.testData = {
     data = torch.Tensor(),
     labels = torch.Tensor(),
     size = function() return testsize end
  }
  self.testData.data, self.testData.labels = parseDataLabel(raw_test.data,
                                                 testsize, channel, height, width)
  local testData = self.testData

      self.unlabelData = {
         data = torch.Tensor(),
         size = function() return unlabelsize end
      }
      self.unlabelData.data = parseUnlabelData(raw_unlabel.data,
                                                         unlabelsize, channel, height, width)

  -- convert from ByteTensor to Float
  self.trainData.data = self.trainData.data:float()
  self.trainData.labels = self.trainData.labels:float()
  self.valData.data = self.valData.data:float()
  self.valData.labels = self.valData.labels:float()
  self.testData.data = self.testData.data:float()
  self.testData.labels = self.testData.labels:float()
  print 'converting unlabel data'
  --collectgarbage()
  collectgarbage()
end

function Provider:normalize()
  ----------------------------------------------------------------------
  -- preprocess/normalize train/val/unlabel sets
  --
  local trainData = self.trainData
  local valData = self.valData
  local unlabelData=self.unlabelData
  local testData=self.testData
  local count=0

  print '<trainer> preprocessing data (color space + normalization)'
  collectgarbage()
  --
   --[[local centroids = x.new(64,768):normal()
   for i = 1,k do
      centroids[i]:div(centroids[i]:norm())
   end
   local filters=centroids
   torch.save('filters',filters)--]]
   
   
  -- preprocess unlabelData
  --[[local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
   for i=1,unlabelsize,unlabel_batchSize do
      --self.unlabelData.data = self.unlabelData.data:narrow(1,i,unlabel_batchSize):float()
      unlabel_d=self.unlabelData.data:narrow(1,i,unlabel_batchSize):float()
      --print (unlabel_d:size())
      --for j=i,i+unlabel_batchSize-1 do
         
         -- rgb -> yuv
      for j=i,i+unlabel_batchSize-1 do
         xlua.progress(j, unlabelData:size())
         local rgb = unlabel_d[j-(unlabel_batchSize * count)]
         local yuv = image.rgb2yuv(rgb)
         -- normalize y locally:
         yuv[1] = normalization(yuv[{{1}}])
         unlabel_d[j-(unlabel_batchSize * count)] = yuv
      end
         -- normalize u globally:
         local mean_u = unlabel_d:select(2,2):mean()
         local std_u = unlabel_d:select(2,2):std()
         unlabel_d:select(2,2):add(-mean_u)
         unlabel_d:select(2,2):div(std_u)
         -- normalize v globally:
         local mean_v = unlabel_d:select(2,3):mean()
         local std_v = unlabel_d:select(2,3):std()
         unlabel_d:select(2,3):add(-mean_v)
         unlabel_d:select(2,3):div(std_v)

         --unlabel_d.mean_u = mean_u
         --unlabel_d.std_u = std_u
         --unlabel_d.mean_v = mean_v
         --unlabel_d.std_v = std_v
         
         --self.unlabelData.data = self.unlabelData.data:narrow(1,i,unlabel_batchSize):ByteTensor()
         
     
      local filters = learn_filters(unlabel_d,count)
      torch.save('filters',filters)
      unlabel_data=nil
      count=count+1
  end--]]
  -- preprocess trainSet
  local normalization = nn.SpatialContrastiveNormalization(1, image.gaussian1D(7))
  for i = 1,trainData:size() do
     xlua.progress(i, trainData:size())
     -- rgb -> yuv
     local rgb = trainData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[1] = normalization(yuv[{{1}}])
     trainData.data[i] = yuv
  end
  -- normalize u globally:
  local mean_u = trainData.data:select(2,2):mean()
  local std_u = trainData.data:select(2,2):std()
  trainData.data:select(2,2):add(-mean_u)
  trainData.data:select(2,2):div(std_u)
  -- normalize v globally:
  local mean_v = trainData.data:select(2,3):mean()
  local std_v = trainData.data:select(2,3):std()
  trainData.data:select(2,3):add(-mean_v)
  trainData.data:select(2,3):div(std_v)

  trainData.mean_u = trainData.mean_u
  trainData.std_u = trainData.std_u
  trainData.mean_v = trainData.mean_v
  trainData.std_v = trainData.std_v

  -- preprocess valSet
  for i = 1,valData:size() do
    xlua.progress(i, valData:size())
     -- rgb -> yuv
     local rgb = valData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     valData.data[i] = yuv
  end
  -- normalize u globally:
  valData.data:select(2,2):add(-mean_u)
  valData.data:select(2,2):div(std_u)
  -- normalize v globally:
  valData.data:select(2,3):add(-mean_v)
  valData.data:select(2,3):div(std_v)

   for i = 1,testData:size() do
     xlua.progress(i, testData:size())
     -- rgb -> yuv
     local rgb = testData.data[i]
     local yuv = image.rgb2yuv(rgb)
     -- normalize y locally:
     yuv[{1}] = normalization(yuv[{{1}}])
     testData.data[i] = yuv
   end
  -- normalize u globally:
  testData.data:select(2,2):add(-mean_u)
  testData.data:select(2,2):div(std_u)
  -- normalize v globally:
  testData.data:select(2,3):add(-mean_v)
  testData.data:select(2,3):div(std_v)
  
  --return testData,trainData,valData
end
