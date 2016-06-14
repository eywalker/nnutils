require 'nn'
local SwitchableContainer, parent = torch.class('nn.SwitchableContainer', 'nn.Sequential')

function SwitchableContainer:__init(trainable)
  parent.__init(self)
  if trainable==nil then trainable=true end
  self.trainable = trainable
end


function SwitchableContainer:accGradParameters(...)
  if self.trainable then
    parent.accGradParameters(self, ...)
  end
end

function SwitchableContainer:backward(...)
  if self.trainable then
    return parent.backward(self, ...)
  else
    return self:updateGradInput(...)
  end
end

function SwitchableContainer:accUpdateGradParameters(...)
  if self.trainable then
    parent.accUpdateGradParameters(self, ...)
  end
end
