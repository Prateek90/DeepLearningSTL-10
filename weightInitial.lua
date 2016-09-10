require 'nn'

do

    local Spatialkmeans, parent = torch.class('nn.Spatialkmeans', 'nn.SpatialConvolution')
   
    -- override the constructor to have the additional range of initialization
    function Spatialkmeans:__init(nInputPlane, nOutputPlane,initialweights, kW, kH, dW, dH, padW, padH)
        parent.__init(self,nInputPlane, nOutputPlane, kW, kH, dW, dH, padW, padH)
               
        self:reset(initialweights)
    end
   
    -- override the :reset method to use custom weight initialization.        
    function Spatialkmeans:reset(initialweights)
       
        if stdv then
          stdv = stdv * math.sqrt(3)
        else
          stdv = 1/math.sqrt(self.kW*self.kH*self.nInputPlane)
        end
        if nn.oldSeed then
          --[[self.weight:apply(function()
          return torch.uniform(-stdv, stdv)
          end)--]]
          if self.bias then
            self.bias:apply(function()
            return torch.uniform(-stdv, stdv)
            end)
          end
        else
          --self.weight:uniform(-stdv, stdv)
          if self.bias then
            self.bias:uniform(-stdv, stdv)
          end
        end
        self.weight=initialweights
    end
end
