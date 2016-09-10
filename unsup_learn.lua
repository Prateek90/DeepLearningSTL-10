require 'unsup'
require 'image'
require 'nn'
require 'torch'

inputsize = 768
outputsize = 16

if params.model == 'linear' then
  
   print "constructing linear model"

   --------constructing Encoder

   encoder = nn.Sequential()
   encoder:add(nn.Linear(inputsize,outputsize))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputsize))

   --------constructing Decoder

   decoder =nn.Sequential()
   decoder:add(nn.Linear(outputsize,inputsize))

   --------complete model

   module=unsup.AutoEncoder(encoder,decoder,params.beta)

   print "constructed Linear AutoEncoder"
   
elseif params.model == 'conv' then
  
   -- params:
   conntable = nn.tables.full(params.nfiltersin, params.nfiltersout)
   kw, kh = params.kernelsize, params.kernelsize
   iw, ih = params.inputs, params.inputs

   -- connection table:
   local decodertable = conntable:clone()
   decodertable[{ {},1 }] = conntable[{ {},2 }]
   decodertable[{ {},2 }] = conntable[{ {},1 }]
   local outputFeatures = conntable[{ {},2 }]:max()

   -- encoder:
   encoder = nn.Sequential()
   encoder:add(nn.SpatialConvolution(3,16, 8, 8, 1, 1))
   encoder:add(nn.Tanh())
   encoder:add(nn.Diag(outputFeatures))

   -- decoder:
   decoder = nn.Sequential()
   decoder:add(nn.SpatialFullConvolution(3,16, 8, 8, 1, 1))

   -- complete model
   module = unsup.AutoEncoder(encoder, decoder, params.beta)

   -- convert dataset to convolutional (returns 1xKxK tensors (3D), instead of K*K (1D))
   --unlabelData:conv()

   -- verbose
   print('==> constructed convolutional auto-encoder')

end

