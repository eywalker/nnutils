nnutils = nnutils or {}

local Tracker = torch.class('nnutils.Tracker')

function Tracker:__init()
  self.epochNumber = 0
  self.tota_samples = 0
  self.loss_epoch = 0
  self.top1_epoch = 0
  self.batchNumber = 0
  self.dataTimer = torch.Timer()
  self.processTimer = torch.Timer()
  self.dataLoadTime = 0
  self.dataProcessTime = 0
end

function Tracker:startEpoch(n)
  self.epochNumber = n or self.epochNumber + 1
  self.batchNumber = 0
  self.dataTimer:reset()
end

function Tracker:startBatch(n)
  self.batchNumber = n or self.batchNumber + 1
  self.dataProcessTime = self.dataTimer:time().real
end

function Tracker:logBatch(arg)
end

function Tracker:endBatch(n)
  self.
end


