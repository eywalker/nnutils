require 'torch'
require 'nnutils.minibatch'
local paths = require 'paths'

nnutils = nnutils or {}

-- Template data loader class
local DataLoader = torch.class('nnutils.DataLoader')

function DataLoader:__init(arg)
  self.train_preproc = {}
  self.test_preproc = {}
  self.nBatches = nil
  self.batchSize = nil
  -- empty trainset
  self.trainsetData = torch.Tensor()
  self.trainsetLabels = torch.Tensor()
  self.trainsetSize = 0
  -- empty testest
  self.testsetData = torch.Tensor()
  self.testsetLabels = torch.Tensor()
  self.testsetSize = 0
  self.epochIndicies = torch.LongTensor()
end

function DataLoader:setTrainset(trainsetData, trainsetLabels)
  assert(trainsetData:size(1) == trainsetLabels:size(1), "Sizes of data and labels don't match!")
  self.trainsetData:resize(trainsetData:size()):copy(trainsetData)
  self.trainsetLabels:resize(trainsetLabels:size()):copy(trainsetLabels)
  self.trainsetSize = trainsetData:size(1)
end

function DataLoader:setTestset(testsetData, testsetLabels)
  assert(testsetData:size(1) == testsetLabels:size(1), "Sizes of data and labels don't match!")
  self.testsetData = testsetData
  self.testsetLabels = testsetLabels
  self.testsetSize = testsetData:size(1)
end

function DataLoader:getTrainset(indicies)
  local data, labels
  if not indicies then
    data = self.trainsetData:clone()
    labels = self.trainsetLabels:clone()
  else
    data = self.trainsetData:index(1, indicies)
    labels = self.trainsetLabels:index(1, indicies)
  end
  return data, labels
end

function DataLoader:getTestset(indicies)
  local data, labels
  if not indicies then
    data = self.testsetData:clone()
    labels = self.testsetLabels:clone()
  else
    data = self.testsetData:index(1, indicies)
    labels = self.testsetLabels:index(1, indicies)
  end
  return data, labels
end

-- randomly sample `quantity` samples from trainset
function DataLoader:sampleTrainset(quantity)
  local indicies = torch.LongTensor(quantity):random(1, self.trainsetSize)
  return self:getTrainset(indicies)
end

function DataLoader:sampleTestset(quantity)
  local indicies = torch.LongTensor(quantity):random(1, self.testsetSize)
  return self:getTestset(indicies)
end

-- Start a new epoch, resetting the batch size
-- and total number of batches
function DataLoader:startEpoch(batchSize, nBatches)
  if (randomize==nil) then randomize = true end
  batchSize = batchSize or 100
  self.batchSize = batchSize
  self.nBatches = math.ceil(self.trainsetSize / self.batchSize)
  if randomize then
    self.epochIndicies:randperm(self.trainsetSize)
  end
end

function DataLoader:getMiniBatch(batchNumber)
  local start = (batchNumber - 1) * self.batchSize + 1
  local stop = math.min((batchNumber) * self.batchSize, self.trainSet:size())
  local batchIndicies = self.epochIndicies[{{start, stop}}]
  local data, labels = self:getTrainset(batchIndicies)
  return MiniBatch(data, labels)
end
