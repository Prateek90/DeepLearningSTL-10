-- get all parameters
---------------------------
dofile './provider.lua'

provider = Provider()
provider:normalize()
torch.save('provider.t7',provider)

local unlabelData = provider.unlabelData.data:float()
---------------------------

x,dl_dx,ddl_ddx = module:getParameters()

local inputs = {}
local targets = {}

-- training errors
local err = 0
local iter = 0
for t = 1,params.maxiter,params.batchsize do
  
  iter = iter+1
  xlua.progress(iter*params.batchsize, params.statinterval)

  --local example = dataset_unlabel[t]
  --local inputs = {}
  --local targets = {}
  --for i = t,t+params.batchsize-1 do
     -- load new sample
  local sample = unlabelData[t]
  sample=sample:float()
  local input = sample[1]:clone()
  local target = sample[2]:clone()
     --print ("input:",input,input:size())
     --print ("target:",target:size())
  table.insert(inputs, input)
  table.insert(targets, target)
  --end
  
  --------------------------------------------------------------------
   -- define eval closure
   --
   local feval = function()
      -- reset gradient/f
      local f = 0
      dl_dx:zero()

      -- estimate f and gradients, for minibatch
      for i = 1,#inputs do
         -- f
         f = f + module:updateOutput(inputs[i], targets[i])

         -- gradients
         module:updateGradInput(inputs[i], targets[i])
         module:accGradParameters(inputs[i], targets[i])
      end

      -- normalize
      dl_dx:div(#inputs)
      f = f/#inputs

      -- return f and df/dx
      return f,dl_dx
   end

--------------------------------------------------------------------
   -- one SGD step
   --
   --[[sgdconf = sgdconf or {learningRate = params.eta,
                         learningRateDecay = params.etadecay,
                         learningRates = etas,
                         momentum = params.momentum}
   _,fs = optim.sgd(feval, x, sgdconf)
   err = err + fs[1]*params.batchsize -- so that err is indep of batch size
   --]]
   -- normalize
   --[[if params.model:find('psd') then
      module:normalize()
   end--]]
end
print (inputs:size())
print (targets:size())
--------------------------------------------------------------------
