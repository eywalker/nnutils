require 'torch'
local threads = require 'threads'

threads.Threads.serialization('threads.sharedserialize')

nnutils = nnutils or {}

local Trainer = torch.class('nnutils.Trainer')

function Trainer:__init(model, criterion, dataLoader)
  self.model = model
  self.criterion = criterion
  self.dataLoader = dataLoader

  self.is_cuda = nil -- starts out not in cuda
  self:float()
end

function Trainer:cuda()
  self.model:cuda()
  self.criterion:cuda()
  self.is_cuda = true
end

function Trainer:float()
  self.model:float()
  self.criterion:float()
  self.is_cuda = false
end

function Trainer:double()
  self.model:double()
  self.criterion:double()
  self.is_cuda = false
end

function Trainer:train(batchSize, eta, nEpochs)
  nEpochs = nEpochs or 1
  eta = eta or 0.01
  local data, labels
  if self.is_cuda then
    data = torch.CudaTensor()
    labels = torch.CudaTensor()
  end
  local model = self.model
  local criterion = self.criterion
  for n=1,nEpochs do
    self.dataLoader:startEpoch(batchSize)
    local nBatches = self.dataLoader:nBatches()
    for i=1,nBatches do
      local data_cpu, labels_cpu = self.dataLoader:getMiniBatch(i)
      if self.is_cuda then
        -- copy data to GPU
        data:resize(data_cpu:size()):copy(data_cpu)
        labels:resize(labels_cpu:size()):copy(labels_cpu)
      else
        -- working on cpu, so simply refer to batch data
        data = data_cpu
        labels = labels_cpu
      end
      local cost = criterion:forward(model:forward(data), labels)
      model:zeroGradParameters()
      model:backward(data, criterion:backward(model.output, labels))
      model:updateParameters(eta)
      print(string.format('Batch %d/%d: cost=%.2f', i, nBatches, cost))
    end
    collectgarbage()
    collectgarbage()
  end
end

