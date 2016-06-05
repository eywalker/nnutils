require 'nnutils'
require 'nn'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')

-- define model...
local model = nn.Sequential()
model:add(nn.View(28*28))
model:add(nn.Linear(28*28, 2048))
model:add(nn.ReLU())
model:add(nn.Linear(2048, 10))
model:add(nn.LogSoftMax())
-- ...and criterion
local criterion = nn.ClassNLLCriterion()

-- initialize the data loader
local mnist = nnutils.MNISTLoader()
-- configure data preprocessor
mnist.trainPC:add(nnutils.RandomCrop(32, 32, 28, 28))

-- initialize trainer
local nThreads = 12
local trainer = nnutils.Trainer(model, criterion, mnist, nThreads)
-- run on GPU!
trainer:cuda()

-- kickoff the training!
trainer:train(80, 600)
