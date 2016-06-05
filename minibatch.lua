-- Given input and labels, construct indexable dataset
-- that conforms to `nn` trainer's requirement.
nnutils = nnutils or {}

local function normalize(self, mean_, std_)
  -- normalize across samples for each pixel/channel separately
  local data = self.data
  local mean = mean_ or data:mean(1)
  -- non-corrected stdard deviation (divide by N, not N-1)
  local std = std_ or data:std(1, true)
  data:add(-mean:expandAs(data))
  local mask = std:lt(1e-12)
  local std_adj = std:clone()
  std_adj:maskedFill(mask,1)
  data:cdiv(std_adj:expandAs(data))
  return mean, std
end

local function normalizeGlobal(self, mean_, std_)
  -- normalize across the entire set of images
  local data = self.data
  local mean = mean_ or data:mean()
  local std = std_ or data:std()
  data:add(-mean)
  data:div(std)
  return mean, std
end

local function size(self)
  return self.data:size(1)
end

-- returns CudaTensor copy of the
-- minibatch dataset
local function cuda(self)
  require 'cutorch'
  local data_cuda = self.data:cuda()
  local labels_cuda = self.labels:cuda()
  return nnutils.MiniBatch(data_cuda, labels_cuda)
end


local function __index(self, index)
  local input = self.data[index]
  local label = self.labels[index]
  return {input, label}
end

-- factory function to return table of data and labels
-- with indexing capacity compatible with trainer native
-- Torch nn package
function nnutils.MiniBatch(data, labels)
  local dataset = {}
  dataset.data = data
  dataset.labels = labels
  dataset.normalize = normalize
  dataset.normalizeGlobal = normalizeGlobal
  dataset.size = size
  dataset.cuda = cuda
  dataset.__index = __index

  setmetatable(dataset, dataset)
  return dataset
end
