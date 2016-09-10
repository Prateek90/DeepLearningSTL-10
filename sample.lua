require 'nn'

local vgg = nn.Sequential()

-- building block
local function ConvBNReLU(nInputPlane, nOutputPlane,layer)
  if laye==1 then
     filters=torch.load('filters')
     filters=torch.reshape(filters,torch.LongStorage(64,3,16,16))
     vgg:add(nn.Spatialkmeans(nInputPlane, nOutputPlane,filters, 3,3, 1,1, 1,1,filters))
     vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
     vgg:add(nn.ReLU(true))
  else
     vgg:add(nn.SpatialConvolution(nInputPlane, nOutputPlane, 3,3, 1,1, 1,1))
     vgg:add(nn.SpatialBatchNormalization(nOutputPlane,1e-3))
     vgg:add(nn.ReLU(true))
  end
  return vgg
end

-- Will use "ceil" MaxPooling because we want to save as much
-- space as we can
local MaxPooling = nn.SpatialMaxPooling

ConvBNReLU(3,64,1)
vgg:add(MaxPooling(4,4,4,4):ceil())

ConvBNReLU(64,128,2)
vgg:add(MaxPooling(3,3,3,3):ceil())

ConvBNReLU(128,256,3)
ConvBNReLU(256,256)
vgg:add(MaxPooling(2,2,2,2):ceil())

ConvBNReLU(256,256,4)
ConvBNReLU(256,256,5)
ConvBNReLU(256,256,6)
vgg:add(MaxPooling(2,2,2,2):ceil())

vgg:add(nn.View(256*2*2))

classifier = nn.Sequential()
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256*2*2,256))
classifier:add(nn.BatchNormalization(256))
classifier:add(nn.ReLU(true))
classifier:add(nn.Dropout(0.5))
classifier:add(nn.Linear(256,10))
vgg:add(classifier)

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'nn.SpatialConvolution'
end

MSRinit(vgg)

-- check that we can propagate forward without errors
-- should get 16x10 tensor
--print(#vgg:cuda():forward(torch.CudaTensor(16,3,32,32)))

return vgg
