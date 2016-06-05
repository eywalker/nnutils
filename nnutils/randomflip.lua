require 'nnutils.processor'
require 'image'

local RandomFlip, parent = torch.class('nnutils.RandomFlip', 'nnutils.Processor')

function RandomFlip:__init(flipDir, flipProbability)
  parent.__init(self)
  flipDir = flipDir or 'h'
  flipDir = flipDir:lower()
  self.flipV = flipDir:find('v') ~= nil
  self.flipH = flipDir:find('h') ~= nil
  self.flipP = flipProbability or 0.5
end

function RandomFlip:updateOutput(data, labels)
  local outData = data:clone()
  local nInputs = outData:size(1)
  for i=1,nInputs do
    if torch.uniform() < self.flipP then
      if self.flipV then image.vflip(outData[i], outData[i]) end
      if self.flipH then image.hflip(outData[i], outData[i]) end
    end
  end
  return outData, labels
end
