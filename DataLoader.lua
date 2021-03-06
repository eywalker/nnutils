require 'torch'
require 'nnutils.processingchain'
require 'nnutils.misc'
local paths = require 'paths'

nnutils = nnutils or {}

-- Base data loader class with essential functionalities such as
-- MiniBatch generation and data preprocessing
local DataLoader = torch.class('nnutils.DataLoader')

function DataLoader:__init(arg)
  arg = arg or {}
  self.trainPC = nnutils.ProcessingChain()
  self.testPC = nnutils.ProcessingChain()
  self._nBatches = torch.Tensor(1)
  self._batchSize = torch.Tensor(1)
  self._nTestBatches = torch.Tensor(1)
  self._testBatchSize = torch.Tensor(1)

  -- empty trainset
  self.trainsetData = torch.Tensor()
  self.trainsetLabels = torch.Tensor()
  self._trainsetSize = torch.Tensor(1)

  -- empty testest
  self.testsetData = torch.Tensor()
  self.testsetLabels = torch.Tensor()
  self._testsetSize = torch.Tensor(1)

  -- epoch indicies
  self.epochIndicies = torch.LongTensor()

  -- batches are drawn with replacement
  self.batchWR = arg.batchWR or false
end


function DataLoader:setTrainset(trainsetData, trainsetLabels)
  assert(trainsetData:size(1) == trainsetLabels:size(1), "Sizes of data and labels don't match!")
  self.trainsetData:resize(trainsetData:size()):copy(trainsetData)
  self.trainsetLabels:resize(trainsetLabels:size()):copy(trainsetLabels)
  self:setTrainsetSize(trainsetData:size(1))
end

function DataLoader:setTestset(testsetData, testsetLabels)
  assert(testsetData:size(1) == testsetLabels:size(1), "Sizes of data and labels don't match!")
  self.testsetData = testsetData
  self.testsetLabels = testsetLabels
  self:setTestsetSize(testsetData:size(1))
end

--------------------- methods to be overloaded by subclasses -------------------
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

function DataLoader:type(typeName)
  self.trainsetData = self.trainsetData:type(typeName)
  self.testsetData = self.testsetData:type(typeName)
end
----------------- end of methods to be overloaded by subclasses ----------------

function DataLoader:cuda()
  self:type('torch.CudaTensor')
end

function DataLoader:float()
  self:type('torch.FloatTensor')
end

function DataLoader:getProcessedTrainset(indicies)
  return self.trainPC:forward(self:getTrainset(indicies))
end

function DataLoader:getProcessedTestset(indicies)
  return self.testPC:forward(self:getTestset(indicies))
end

-- randomly sample `quantity` samples from trainset
function DataLoader:sampleTrainset(quantity)
  local indicies = torch.LongTensor(quantity):random(1, self:getTrainsetSize())
  return self:getProcessedTrainset(indicies)
end

function DataLoader:sampleTestset(quantity)
  local indicies = torch.LongTensor(quantity):random(1, self:getTestsetSize())
  return self:getProcessedTestset(indicies)
end

-- Start a new epoch, resetting the batch size
-- and total number of batches
function DataLoader:startEpoch(batchSize, nBatches)
  batchSize = batchSize or math.min(100, self:getTrainsetSize())
  self._batchSize[1] = batchSize
  local fullBatches = math.ceil(self:getTrainsetSize() / self:batchSize())
  nBatches = nBatches or fullBatches
  if not self.batchWR then nBatches = math.min(nBatches, fullBatches) end
  self._nBatches[1] = nBatches
  self.epochIndicies:randperm(self:getTrainsetSize())
  return nBatches
end

function DataLoader:getMiniBatch(batchNumber)
  if self.batchWR then
    return self:sampleTrainset(self:batchSize())
  end
  local start = (batchNumber - 1) * self:batchSize() + 1
  local stop = math.min((batchNumber) * self:batchSize(), self:getTrainsetSize())
  local batchIndicies = self.epochIndicies[{{start, stop}}]
  return self:getProcessedTrainset(batchIndicies)
end

function DataLoader:setTestBatches(batchSize)
  batchSize = batchSize or math.min(100, self:getTestsetSize())
  self._testBatchSize[1] = batchSize
  local nBatches = math.ceil(self:getTestsetSize() / self:testBatchSize())
  self._nTestBatches[1] = nBatches
  return nBatches
end

function DataLoader:getTestBatch(batchNumber)
  local start = (batchNumber - 1) * self:testBatchSize() + 1
  local stop = math.min((batchNumber) * self:testBatchSize(), self:getTestsetSize())
  local batchIndicies = torch.LongTensor():range(start, stop)
  return self:getProcessedTestset(batchIndicies)
end

---------------- setters and getters --------------------------
function DataLoader:getTrainsetSize()
  return self._trainsetSize[1]
end

function DataLoader:setTrainsetSize(n)
  self._trainsetSize[1] = n
end

function DataLoader:getTestsetSize()
  return self._testsetSize[1]
end

function DataLoader:setTestsetSize(n)
  self._testsetSize[1] = n
end

function DataLoader:batchSize()
  return self._batchSize[1]
end

function DataLoader:testBatchSize()
  return self._testBatchSize[1]
end

function DataLoader:nBatches()
  return self._nBatches[1]
end

function DataLoader:nTestBatches()
  return self._nTestBatches[1]
end
