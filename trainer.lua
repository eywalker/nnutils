require 'torch'
require 'optim'
require 'nnutils.misc'
local argcheck = require 'argcheck'
local threads = require 'threads'

threads.Threads.serialization('threads.sharedserialize')

nnutils = nnutils or {}

local Trainer = torch.class('nnutils.Trainer')


function Trainer:__init(model, criterion, dataLoader, nThreads, verbosity)
  self.dataLoader = dataLoader
  self.nThreads = nThreads or 1
  self:setupPool()

  self.verbosity = verbosity or 2  -- defalut to maximum verbosity

  self.model = model
  self.parameters = nil
  self.gradParameters = nil
  self.criterion = criterion

  self.data = torch.Tensor()
  self.targets = torch.Tensor()

  self.optimState = {
    learningRate = 0.01,
    learningRateDecay = 0.0,
    momentum = 0.0,
    dampening = 0.0,
    weightDecay = 0.0
  }

  -- initialize to float
  self:float()
  self.dataLoader:float()

  self.dataTimer = torch.Timer() -- measure time to load data
  self.timer = torch.Timer() -- measure time to complete minibatch
end


-- Configure the threadpool and initialize them. If nThreads == 1, then
-- it skips setting up the pool and sets global variable instead. This
-- is useful for debugging purpose but generally not recommended.
function Trainer:setupPool()
  if self.nThreads > 1 then
    self.pool = threads.Threads(
      self.nThreads,
      function()
        require 'nnutils'
      end,
      function(threadid)
        _dataLoader = self.dataLoader
        print(string.format('Starting minion with id; %d', threadid))
      end
    )
  else
    -- grudgingly add global variable
    _dataLoader = self.dataLoader
    local pool = {}
    function pool:addjob(f1, f2) f2(f1()) end
    function pool:synchronize() end
    self.pool = pool
  end
end

function Trainer:getParameters()
  local parameters, gradParameters = self.model:getParameters()
  self.parameters = parameters
  self.gradParameters = gradParameters
end

function Trainer:cuda()
  self:type('torch.CudaTensor')
end

function Trainer:float()
  self:type('torch.FloatTensor')
end

function Trainer:type(typeName)
  self.tensorType = typeName
  self.model:type(typeName)
  self.criterion:type(typeName)
  self:getParameters()
  self.useCUDA = typeName == 'torch.CudaTensor'
  if useCUDA then require 'cunn' end
  self.data = self.data:type(typeName)
  self.targets = self.targets:type(typeName)
end


-- initialize member properties at the beginning
-- of a training epoch
function Trainer:startEpoch(n, batchSize, nBatches)
  self.epoch = n
  self.total_samples = 0
  self.loss_epoch = 0
  self.top1_epoch = 0
  self.batchNumber = 0
  self.batchSize = batchSize
  self.nBatches = self.dataLoader:startEpoch(batchSize, nBatches)
  self.dataTimer:reset()
  return self.nBatches
end

function Trainer:startTest(batchSize)
  self.total_test_samples = 0
  self.test_loss = 0
  self.top1_test = 0
  self.testBatchNumber = 0
  self.testBatchSize = batchSize
  self.nTestBatches = self.dataLoader:setTestBatches(batchSize)
  return self.nTestBatches
end


function Trainer:train(nEpochs, batchSize, nBatches)
  nEpochs = nEpochs or 1
  for n=1,nEpochs do
    -- train
    local nBatches = self:startEpoch(n, batchSize, nBatches)
    for i=1, nBatches do
      self.pool:addjob(
        function() return _dataLoader:getMiniBatch(i) end,
        function(...) self:trainBatch(...) end)
    end
    self.pool:synchronize()
    self:cudasync()

    -- save model
    self:saveModel()

    -- test
    local nTestBatches = self:startTest(batchSize)
    for i=1, nTestBatches do
      self.pool:addjob(
        function() return _dataLoader:getTestBatch(i) end,
        function(...) self:testBatch(...) end)
    end
    self.pool:synchronize()
    self:cudasync()
  end
end

function Trainer:saveModel()
  -- stub method: save model to disk
end

function Trainer:trainBatch(data_cpu, targets_cpu)
  local batchSize = data_cpu:size(1)

  self:cudasync()
  collectgarbage()
  local dataLoadingTime = self.dataTimer:time().real
  self.timer:reset()

  local model = self.model
  local criterion = self.criterion
  local data = self.data
  local targets = self.targets
  if self.useCUDA then
    -- transfer to GPU
    data:resize(data_cpu:size()):copy(data_cpu)
    targets:resize(targets_cpu:size()):copy(targets_cpu)
  else
    -- reference passing on CPU
    data = data_cpu:type(self.tensorType)
    targets = targets_cpu:type(self.tensorType)
  end

  local err, outputs
  feval = function(x)
    model:zeroGradParameters()
    outputs = model:forward(data)
    err = criterion:forward(outputs, targets)
    local gradOutputs = criterion:backward(outputs, targets)
    model:backward(data, gradOutputs)
    return err, self.gradParameters
  end
  -- perform single step SGD
  optim.sgd(feval, self.parameters, self.optimState)

  self:cudasync()
  self.batchNumber = self.batchNumber + 1
  self.loss_epoch = self.loss_epoch + err

  local processingTime = self.timer:time().real
  -- top-1 error
  local top1 = nnutils.topkScore(outputs:float(), targets_cpu, 1)
  self.top1_epoch = self.top1_epoch + top1
  self.total_samples = self.total_samples + batchSize
  top1 = top1 * 100 / batchSize

  -- print information for this batch
  print(string.format(
        'Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DataLoadingTime %.3f',
         self.epoch, self.batchNumber, self.nBatches, processingTime, err, top1,
         self.optimState.learningRate, dataLoadingTime))
  self.dataTimer:reset()
end

function Trainer:testBatch(data_cpu, targets_cpu)
  local batchSize = data_cpu:size(1)
  self:cudasync()
  collectgarbage()

  local model = self.model
  local criterion = self.criterion
  local data = self.data
  local targets = self.targets
  if self.useCUDA then
    -- transfer to GPU
    data:resize(data_cpu:size()):copy(data_cpu)
    targets:resize(targets_cpu:size()):copy(targets_cpu)
  else
    -- reference passing on CPU
    data = data_cpu:type(self.tensorType)
    targets = targets_cpu:type(self.tensorType)
  end

  local outputs = model:forward(data)
  local err = criterion:forward(outputs, targets)
  self:cudasync()

  self.testBatchNumber = self.testBatchNumber + 1
  self.test_loss = self.test_loss + err
  local top1 = nnutils.topkScore(outputs:float(), targets_cpu, 1)
  self.total_test_samples = self.total_test_samples + batchSize
  self.top1_test = self.top1_test + top1
  top1 = top1 * 100 / batchSize

  print(string.format(
        'Epoch: Testing [%d][%d/%d]\t Err %.4f Top1-%%: %.2f',
         self.epoch, self.testBatchNumber, self.nTestBatches, err, top1))
end

-- synchronize cutorch only if CUDA is enabled
function Trainer:cudasync()
  if self.useCUDA then cutorch.synchronize() end
end
