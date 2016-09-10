require 'unsup'
require 'image'
require 'optim'

cmd = torch.CmdLine()
cmd:text('Training a simple Linear AutoEncoder on STL-10 images')
cmd:text()
cmd:text('Options')
--general options

cmd:option('-inputsize', 25, 'size of each input patch')
cmd:option('-nfeats', 3, 'number of input filters')
cmd:option('-nfiltersin', 3, 'number of input filters')
cmd:option('-nfiltersout', 16, 'number of input filters')
cmd:option('-nkernelsize',8, 'size of kernel')
cmd:option('-ninputs', 96,'size of input image')
cmd:option('-model', 'conv', 'type of model')
cmd:option('-lambda', 1, 'sparsity coefficient')
cmd:option('-beta', 1, 'prediction error coefficient')
cmd:option('-eta', 2e-3, 'learning rate')
cmd:option('-batchsize', 1, 'batch size')
cmd:option('-unlabelbatchsize', 5000, 'batch size of unlabeled data')
cmd:option('-etadecay', 1e-5, 'learning rate decay')
cmd:option('-maxiter', 10000, 'maximum number of iteration')
cmd:option('-statinterval', 5000, 'interval for saving stats and models')

-- for linear model only:
cmd:option('-tied', false, 'decoder weights are tied to encoder\'s weights (transposed)')

params = cmd:parse(arg or {})


--------------creating patches

--dofile 'patch_extract.lua'
--dofile 'trainData_filters.lua'
--print "Training Complete"

--dataset_unlabel = learn_filters()

--[[dofile './provider.lua'

provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)--]]


--------------creating model

--dofile 'unsup_learn.lua'
dofile 'sample.lua'

--------------Training the model

dofile 'train.lua'

--dofile 'test.lua'
--test()

-----------
print "Full training complete"

--weight=module:getParameters()
--weight=weight:narrow(1,1,12288)

--dataset_train=learn_filters(trainData)

--dofile 'train.lua'

--print (weight)
