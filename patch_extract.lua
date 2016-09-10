require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'unsup'

--[[dofile './provider.lua'

provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)--]]


--number of dimensions
k=64
nfeats = 3
--width and height of the image
width = 96
height = 96

patch_size=16
stride=8
num_patches = (width-patch_size)/stride + 1 
total_patch = num_patches * num_patches

function patches(image_vector)
 
  --print (image_vector:size())
  patch = torch.FloatTensor(total_patch, nfeats, patch_size ,patch_size)
  --print (patch:size())
  local count_patch = 1   

  for w = 1, width-patch_size + 1, stride do
      for ht = 1, height-patch_size + 1, stride do
          for dim = 1, nfeats do
   	        patch[{count_patch,dim}]=image_vector[{dim,{w, w+patch_size-1},{ht, ht+patch_size-1}}]:clone()
	  end
	  count_patch = count_patch + 1
      end
   end
   patch = torch.reshape(patch, torch.LongStorage{total_patch , nfeats * patch_size * patch_size})

   return patch

end

--[[function patches_preprocess(patch,mean_unsup,std)

   for dim = 1,nfeats do
      for ph = 1,patch:size(2) do
         patch[{dim,ph}]:add(-mean_unsup)
	 patch[{dim,ph}]:div(std)
      end
   end

   return patch

end--]]
   
function unsup_transform(image_vector)
   X = patches(image_vector)
   --X = patches_preprocess(X,mean_unsup,std)
   return X
end

--build our data matrix for the supervised parts of algorithm
function learn_filters(unlabelData,count)
   --provider = torch.load './provider.t7'
   --provider.unlabelData.data=provider.unlabelData:float()
   
   --local unlabelData = provider.unlabelData.data:float()
   --print (unlabelData)
   local patch_data = torch.FloatTensor(unlabelData:size(1),total_patch, nfeats * patch_size * patch_size):fill(0)
   --get all the normalized patches
   for i =1,unlabelData:size(1) do 
      patch_data[i] = unsup_transform(unlabelData[i])
   end
   size=unlabelData:size(1)
   unlabelData=nil
   
   
   collectgarbage()
   patch_data = torch.reshape(patch_data,torch.LongStorage{size*total_patch,nfeats*patch_size*patch_size})
   if count==0 then
   	local centroids = patch_data.new(64,768):normal()
   	for i = 1,k do
	   centroids[i]:div(centroids[i]:norm())
   	end
   	print (centroids:size())
   	local filters=centroids
   	torch.save('filters',filters)
   end
   local filters = kmeans(unsup.zca_whiten(patch_data),k)
   patch_data=nil
   --filters=torch.reshape(filters,torch.LongStorage{192,16,16})
   --torch.save('filters',filters)
   print "Patch extraction complete"
   --print (patch_data:size())
   --local trData=provider.trainData.data:float()
   --local trlabel=provider.trainData.labels:float()
   --local valData=provider.valData.data:float()
   --return patch_data,trData,valData
   
   --print(filters:size())
   --gfx.image(filters:view(64*3,16,16),{zoom=5.0})
   return filters;

end







