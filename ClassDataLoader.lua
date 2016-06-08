require 'torch'
require 'nnutils.misc'
local paths = require 'paths'

nnutils = nnutils or {}

-- Base data loader class with essential functionalities such as
-- MiniBatch generation and data preprocessing
local ClassDataLoader,parent = torch.class('nnutils.ClassDataLoader', 'nnutils.DataLoader')

function ClassDataLoader:__init(arg)
  arg = arg or {}
  self.balancedRand = arg.balancedRand or false
  if self.balancedRand then arg.batchWR = true end
  parent.__init(self, arg)

  self.nClasses = 0
  self.trainClassIdxList = {}
  self.testClassIdxList = {}
end

local function indexClasses(labels)
  local classIdx = {}
  for i=1,labels:size(1) do
    local indicies = classIdx[labels[i]] or {}
    table.insert(indicies, i)
    classIdx[labels[i]] = indicies
  end

  for k,v in pairs(classIdx) do
    classIdx[k] = torch.LongTensor(v)
  end

  return classIdx
end

function ClassDataLoader:nClasses()
  return self.nClasses
end

function ClassDataLoader:trainClassIndexList(classID)
  return self.trainClassIdxList[classID]
end

function ClassDataLoader:testClassIndexList(classID)
  return self.testClassIdxList[classID]
end


function ClassDataLoader:setTrainset(trainsetData, trainsetLabels)
  parent.setTrainset(self, trainsetData, trainsetLabels)
  self.trainClassIdxList = indexClasses(trainsetLabels)
  -- TODO: think about a better way to obtain number of classes
  self.nClasses = #self.trainClassIdxList
end

function ClassDataLoader:setTestset(testsetData, testsetLabels)
  parent.setTestset(self, testsetData, testsetLabels)
  self.testClassIdxList = indexClasses(testsetLabels)
end

local function drawFrom(values, quantity)
  quantity = quantity or 1
  local draws = torch.LongTensor(quantity)
  local N = values:size(1)
  draws:random(1, N)
  return values:index(1, draws)
end

function ClassDataLoader:trainDrawFromClass(classID, quantity)
  return drawFrom(self:trainClassIndexList(classID), quantity)
end

function ClassDataLoader:testDrawFromClass(classID, quantity)
  return drawFrom(self:testClassIndexList(classID), quantity)
end

-- randomly sample `quantity` samples from trainset
function ClassDataLoader:sampleTrainset(quantity)
  local indicies = torch.LongTensor(quantity)
  if self.balancedRand then
    local classes = torch.LongTensor(quantity):random(1, self.nClasses)
    for i=1,quantity do
      indicies[i] = self:trainDrawFromClass(classes[i])
    end
  else
    indicies = indicies:random(1, self:getTrainsetSize())
  end
  return self:getProcessedTrainset(indicies)
end

function ClassDataLoader:sampleTestset(quantity)
  local indicies = torch.LongTensor(quantity)
  if self.balancedRand then
    local classes = torch.LongTensor(quantity):random(1, self.nClasses)
    for i=1,quantity do
      indicies[i] = self:testDrawFromClass(classes[i])
    end
  else
    indicies = indicies:random(1, self:getTestsetSize())
  end
  return self:getProcessedTestset(indicies)
end

