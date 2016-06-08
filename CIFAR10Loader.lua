--require 'nnutils.dataloader'
require 'nnutils.DataLoader'
require 'nnutils.onehot'
local paths = require 'paths'

local CIFAR10Loader, parent = torch.class('nnutils.CIFAR10Loader', 'nnutils.ClassDataLoader')

function CIFAR10Loader:__init(arg)
  arg = arg or {}
  parent.__init(self, arg)
  self.data_path = arg.data_path or '.' -- defaults to current directory
  if arg.onehot==nil then arg.onehot = false end
  self.onehot = arg.onehot
  if not paths.dirp(self.data_path) then
    assert(paths.mkdir(self.data_path), 'Failed to create directory '..self.data_path)
  end
  self.path_remote = arg.path_remote or 'http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar-10-torch.tar.gz'
  self.path_dataset = paths.concat(self.data_path, 'cifar-10batches-t7')
  self.path_trainset = paths.concat(self.path_dataset, 'data_batch_%d.t7')
  self.path_testset = paths.concat(self.path_dataset, 'test_batch.t7')

  self:setTrainset(self:loadDataset(self.path_trainset, 5))
  self:setTestset(self:loadDataset(self.path_testset, 1))
end


-- Download CIFAR10 dataset into the data_path
function CIFAR10Loader:download()
  if not paths.filep(self.path_trainset) or not paths.filep(self.path_testset) then
      local remote = self.path_remote
      local tmp_dir = paths.tmpname()
      assert(paths.mkdir(tmp_dir))
      local tar = paths.concat(tmp_dir, paths.basename(remote))
      os.execute('wget -P '..tmp_dir..' '..remote..'; '..'tar xvf '..tar..' -C '..self.data_path)
      assert(paths.rmall(tmp_dir, 'yes'))
  end
end

function CIFAR10Loader:loadDataset(fileName, nFiles)
  self:download()

  local all_data = {}

  for i=1,nFiles do
    local f = torch.load(string.format(fileName, i))


  local f = torch.load(fileName, 'ascii')
  local data = f.data:type(torch.getdefaulttensortype())
  local labels = f.labels

  local nExample = data:size(1)
  if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<mnist> loading only ' .. nExample .. ' examples')
  end
  data = data[{{1,nExample},{},{},{}}]
  labels = labels[{{1,nExample}}]
  print('<mnist> done')

  -- use onehot encoding
  if self.onehot then
   labels = nnutils.onehot(labels, 10)
  end

  return data, labels
end
