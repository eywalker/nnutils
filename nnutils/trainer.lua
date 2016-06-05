require 'torch'

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
  self.is_cuda = true
end

function Trainer:train(batchSize, eta, nEpochs)
  nEpochs = nEpochs or 1
  eta = eta or 0.01
  local model = self.model
  local criterion = self.criterion
  for n=1,nEpochs do
    self.dataLoader:startEpoch(batchSize)
    for i=1,self.dataLoader.nBatches do
      local minibatch = self.dataLoader:getMiniBatch(i)
      if self.is_cuda then minibatch = minibatch:cuda() end
      local cost = criterion:forward(model:forward(minibatch.data), minibatch.labels)
      model:zeroGradParameters()
      model:backward(minibatch.data, criterion:backward(model.output, minibatch.labels))
      model:updateParameters(eta)
      print(string.format('Batch %d/%d: cost=%.2f', i, self.dataLoader.nBatches, cost))
    end
  end
end

