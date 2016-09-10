require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

width=96
height=96
nfeats=3
patch_size=16
stride = 8
totalPatches = ((width - patch_size)/stride + 1) * ((height - patch_size)/stride + 1)
numPatch=((width-patch_size)/stride + 1)


function extract_all_patches(image_vector)
   local patches = torch.FloatTensor(totalPatches,nfeats,patch_size, patch_size):fill(0)
   local n = 1
   for w = 1, width-patch_size + 1, stride do
      for h = 1, height-patch_size + 1, stride do
         for chan = 1,3 do
            patches[{n,chan}] = image_vector[{chan,{w, w+ patch_size -1},{h, h + patch_size -1}}]:clone()
         end
         n = n+1
      end
   end
   -- Coerce patches into correct dimensionality --6561 * (3*16*16)
   patches = torch.reshape(patches,torch.LongStorage{totalPatches,nfeats * patch_size * patch_size})
   return patches
end

function X_standardize(X)
   --divide by max, subtract mean
   for i=1,X:size()[1] do
      local row = X[i]
      local abs_row = torch.abs(row)
      local div_value = abs_row:max()
      if div_value >0 then 
         row:div(div_value)
      end
      local mean = row:mean()
      row:add(-mean)
   end
   return unsup.zca_whiten(X)
end

--assuming dim of centroids is k * ndims
--need to take this function and bring back the channels for convolution
function conv_centers(X, centroids)
   --multiplication in Torch
   return torch.mm(X,centroids:transpose(1,2))
   --this is N_patches * k
end

function image_transform(image_vector,filters)
   local X = extract_all_patches(image_vector):clone()
   X = X_standardize(X)
   local X_maps = conv_centers(X,filters):transpose(1,2)
   return torch.reshape(X_maps, torch.LongStorage{64,numPatch,numPatch})
end


function transform_data(trainData,trlabels,filters)
   local data = torch.FloatTensor(trainData:size(1),k,numPatch,numPatch):fill(0)
   for i =1,50 do 
      data[i] = image_transform(trainData[i],filters)
      --print (i)
   end
   return {
   --reshape and then fix the row-major to col-major last columns
   data = data,
   labels = trlabels,
   size = function() return trainData:size(1) end
   }
end

-- transform train data
--if not (paths.filep('transformed_t_data')) then
   dofile 'patch_extract.lua'
   filts,trData,trlabels=learn_filters()
   --local trData=provider.trainData.data:float()
   trainData = transform_data(trData,trlabels,filts)
   torch.save('transformed_t_data',trainData)
--else
--   trainData = torch.load('transformed_t_data')
--end


-- transform test data
--[[if not (paths.filep('transformed_test_data')) then
   testData = transform_data(testData,filts)
   torch.save('transformed_test_data',testData)
else
   testData = torch.load('transformed_test_data')
end--]]
