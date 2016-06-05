require 'nnutils.processor'
require 'image'

local CenterCrop, parent = torch.class('nnutils.CenterCrop', 'nnutils.Processor')

function CenterCrop:__init(inputWidth, inputHeight, outputWidth, outputHeight)
  parent.__init(self)
  self.iW = inputWidth
  self.iH = inputHeight
  self.oW = outputWidth
  self.oH = outputHeight
  self.w1 = math.ceil((self.oW - self.iW)/2)
  self.h1 = math.ceil((self.oH - self.iH)/2)
end

function CenterCrop:updateOutput(data, labels)
  local dimW = data:dim()
  local dimH = dimW - 1
  local size = data:size()
  size[dimW] = self.oW
  size[dimH] = self.oH
  local outData = data.new():resize(size)
  local nInputs = data:size(1)
  for i=1,nInputs do
    outData[i] = image.crop(data[i], self.w1, self.h1, self.w1+self.oW, self.h1+self.oH)
  end
  return outData, labels
end
