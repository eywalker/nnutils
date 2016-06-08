--require 'nnutils.dataloader'
require 'nnutils.DataLoader'
require 'nnutils.onehot'
local paths = require 'paths'

local MNISTLoader, parent = torch.class('nnutils.MNISTLoader', 'nnutils.ClassDataLoader')

function MNISTLoader:__init(arg)
  arg = arg or {}
  parent.__init(self, arg)
  self.data_path = arg.data_path or '.' -- defaults to current directory
  if arg.onehot==nil then arg.onehot = false end
  self.onehot = arg.onehot
  if not paths.dirp(self.data_path) then
    assert(paths.mkdir(self.data_path), 'Failed to create directory '..self.data_path)
  end
  self.path_remote = 'https://s3.amazonaws.com/torch7/data/mnist.t7.tgz'
  self.path_dataset = paths.concat(self.data_path, 'mnist.t7')
  self.path_trainset = paths.concat(self.path_dataset, 'train_32x32.t7')
  self.path_testset = paths.concat(self.path_dataset, 'test_32x32.t7')

  self:setTrainset(self:loadDataset(self.path_trainset))
  self:setTestset(self:loadDataset(self.path_testset))
end


-- Download MNIST dataset into the data_path
function MNISTLoader:download()
  if not paths.filep(self.path_trainset) or not paths.filep(self.path_testset) then
      local remote = self.path_remote
      local tmp_dir = paths.tmpname()
      assert(paths.mkdir(tmp_dir))
      local tar = paths.concat(tmp_dir, paths.basename(remote))
      os.execute('wget -P '..tmp_dir..' '..remote..'; '..'tar xvf '..tar..' -C '..self.data_path)
      assert(paths.rmall(tmp_dir, 'yes'))
  end
end

function MNISTLoader:loadDataset(fileName, maxLoad)
  self:download()

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
