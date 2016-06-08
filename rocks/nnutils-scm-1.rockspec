package = "nnutils"
version = "scm-1"
source = {
   url = "git://github.com/eywalker/nnutils"
}
description = {
   license = "MIT"
}
dependencies = {}
build = {
   type = "builtin",
   modules = {
      nnutils = "init.lua",
      ["nnutils.DataLoader"] = "DataLoader.lua",
      ["nnutils.ClassDataLoader"] = "ClassDataLoader.lua",
      ["nnutils.minibatch"] = "minibatch.lua",
      ["nnutils.MNISTLoader"] = "MNISTLoader.lua",
      ["nnutils.onehot"] = "onehot.lua",
      ["nnutils.processingchain"] = "processingchain.lua",
      ["nnutils.processor"] = "processor.lua",
      ["nnutils.randomcrop"] = "randomcrop.lua",
      ["nnutils.centercrop"] = "centercrop.lua",
      ["nnutils.randomflip"] = "randomflip.lua",
      ["nnutils.trainer"] = "trainer.lua",
      ["nnutils.misc"] = "misc.lua",
   }
}