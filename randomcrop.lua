require 'nnutils.processor'
require 'image'

local RandomCrop, parent = torch.class('nnutils.RandomCrop', 'nnutils.Processor')

function RandomCrop:__init(inputWidth, inputHeight, outputWidth, outputHeight)
  parent.__init(self)
  self.iW = inputWidth
  self.iH = inputHeight
  self.oW = outputWidth
  self.oH = outputHeight
end

function RandomCrop:updateOutput(data, labels)
  local dimW = data:dim()
  local dimH = dimW - 1
  local size = data:size()
  size[dimW] = self.oW
  size[dimH] = self.oH
  local outData = data.new():resize(size)
  local nInputs = data:size(1)
  for i=1,nInputs do
    local h1 = math.ceil(torch.uniform(1e-2, self.iH-self.oH))
    local w1 = math.ceil(torch.uniform(1e-2, self.iH-self.oH))
    outData[i] = image.crop(data[i], w1, h1, w1+self.oW, h1+self.oH)
  end
  return outData, labels
end
