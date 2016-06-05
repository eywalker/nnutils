require 'torch'
require 'nn'
require 'optim'
local threads = require 'threads'

threads.Threads.serialization('threads.sharedserialize')

nnutils = nnutils or {}

local Trainer = torch.class('nnutils.Trainer')

function Trainer:__init(model, criterion, dataLoader, nMinions)
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
  self.nMinions = nMinions or 10

  -- initialize to float
  self:float()

  self:setupPool()
end

function Trainer:setupPool()
  self.pool = threads.Threads(
    self.nMinions,
    function()
      require 'nnutils'
    end,
    function(threadid)
      dataLoader = self.dataLoader
      print(string.format('Starting minion with id; %d', threadid))
    end
  )
end

function Trainer:getParameters()
  local parameters, gradParameters = self.model:getParameters()
  self.parameters = parameters
  self.gradParameters = gradParameters
end

function Trainer:cuda()
  require 'cutorch'
  require 'cunn'
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


function Trainer:train(batchSize, nEpochs)
  self.dataTimer = torch.Timer() -- measure time to load data
  self.timer = torch.Timer() -- measure time to complete minibatch
  nEpochs = nEpochs or 1
  local data, labels
  if self.is_cuda then
    self.data = torch.CudaTensor()
    self.labels = torch.CudaTensor()
  end
  local model = self.model
  local criterion = self.criterion
  for n=1,nEpochs do
    self.epoch = n
    self.loss_epoch = 0.0
    self.batchNumber = 0
    self.dataLoader:startEpoch(batchSize)
    self.nBatches = self.dataLoader:nBatches()

    self.dataTimer:reset()
    for i=1,self.nBatches do
      self.pool:addjob(
        function()
          return dataLoader:getMiniBatch(i)
        end,
        function(...)
          self:trainBatch(...)
        end
      )
    end
    self.pool:synchronize()
    if self.is_cuda then cutorch.synchronize() end
    self:saveModel()
  end
end

function Trainer:saveModel()
  -- stub method: save model to disk
end

function Trainer:trainBatch(data_cpu, labels_cpu)
  if self.is_cuda then cutorch.synchronize() end
  collectgarbage()
  local dataLoadingTime = self.dataTimer:time().real
  self.timer:reset()

  local model = self.model
  local criterion = self.criterion
  local data = self.data
  local labels = self.labels
  if self.is_cuda then
    -- transfer to GPU
    data:resize(data_cpu:size()):copy(data_cpu)
    labels:resize(labels_cpu:size()):copy(labels_cpu)
  else
    -- reference passing on CPU
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
  -- perform single step SGD
  optim.sgd(feval, self.parameters, self.optimState)

  if self.is_cuda then cutorch.synchronize() end
  self.loss_epoch = self.loss_epoch + err
  self.batchNumber = self.batchNumber + 1
  print(string.format(
        'Epoch: [%d][%d/%d]\tTime %.3f Err %.4f DataLoadingTime %.3f',
         self.epoch, self.batchNumber, self.nBatches, self.timer:time().real, err, dataLoadingTime))
  self.dataTimer:reset()
end

