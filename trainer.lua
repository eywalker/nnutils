require 'torch'
require 'optim'
local threads = require 'threads'

threads.Threads.serialization('threads.sharedserialize')

nnutils = nnutils or {}

local Trainer = torch.class('nnutils.Trainer')

function Trainer:__init(model, criterion, dataLoader)
  self.model = model
  self.parameters = nil
  self.gradParameters = nil
  self.criterion = criterion
  self.dataLoader = dataLoader
  self.optimState = {
    learningRate = 0.01,
    learningRateDecay = 0.0,
    momentum = 0.0,
    dampening = 0.0,
    weightDecay = 0.0
  }

  -- initialize to float
  self:float()
end

function Trainer:getParameters()
  local parameters, gradParameters = self.model:getParameters()
  self.parameters = parameters
  self.gradParameters = gradParameters
end

function Trainer:cuda()
  self.model:cuda()
  self.criterion:cuda()
  self:getParameters()
  self.is_cuda = true
end

function Trainer:float()
  self.model:float()
  self.criterion:float()
  self:getParameters()
  self.is_cuda = false
end


function Trainer:train(batchSize, eta, nEpochs)
  nEpochs = nEpochs or 1
  eta = eta or 0.01
  local data, labels
  if self.is_cuda then
    self.data = torch.CudaTensor()
    self.labels = torch.CudaTensor()
  end
  local model = self.model
  local criterion = self.criterion
  for n=1,nEpochs do
    self.loss_epoch = 0.0
    self.dataLoader:startEpoch(batchSize)
    local nBatches = self.dataLoader:nBatches()
    for i=1,nBatches do
      local data_cpu, labels_cpu = self.dataLoader:getMiniBatch(i)
      local err = self:trainBatch(data_cpu, labels_cpu)
      print(string.format('Batch %d/%d: cost=%.2f', i, nBatches, err))
    end
  end
end

function Trainer:trainBatch(data_cpu, labels_cpu)
  if self.is_cuda then cutorch.synchronize() end
  collectgarbage()
  local model = self.model
  local criterion = self.criterion
  local data = self.data
  local labels = self.labels
  if self.is_cuda then
    data:resize(data_cpu:size()):copy(data_cpu)
    labels:resize(labels_cpu:size()):copy(labels_cpu)
  else
    data = data_cpu
    labels = labels_cpu
  end

  local err, outputs
  feval = function(x)
    model:zeroGradParameters()
    outputs = model:forward(data)
    err = criterion:forward(outputs, labels)
    local gradOutputs = criterion:backward(outputs, labels)
    model:backward(data, gradOutputs)
    return err, self.gradParameters
  end
  optim.sgd(feval, self.parameters, self.optimState)

  if self.is_cuda then cutorch.synchronize() end
  self.loss_epoch = self.loss_epoch + err
  return err
end

