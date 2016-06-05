nnutils = nnutils or {}

local ProcessingChain = torch.class('nnutils.ProcessingChain')

function ProcessingChain:__init()
  self.processors = {}
end

function ProcessingChain:add(processor)
  table.insert(self.processors, processor)
end

function ProcessingChain:get(index)
  return self.processors[index]
end

function ProcessingChain:size()
  return #self.processors
end

function ProcessingChain:forward(data, labels)
  local currentData = data
  local currentLabels = labels
  for i=1,#self.processors do
    currentData, currentLabels = self.processors[i]:forward(currentData, currentLabels)
  end
  return currentData, currentLabels
end
