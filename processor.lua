nnutils = nnutils or {}

local Processor = torch.class('nnutils.Processor')

function Processor:__init()
  self.save = false
end

function Processor:forward(data, labels)
  local procData, procLabels = self:updateOutput(data, labels)
  return procData, procLabels
end

function Processor:updateOutput(data, labels)
  return data, labels
end
