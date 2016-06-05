package = "nnutils"
version = "scm-1"
source = {
   url = "https://github.com/eywalker/nnutils"
}
description = {
   license = "MIT"
}
dependencies = {
}
build = {
   type = "builtin",
   modules = {
      nnutils = "init.lua",
      ["nnutils.dataloader"] = "nnutils/dataloader.lua",
      ["nnutils.minibatch"] = "nnutils/minibatch.lua",
      ["nnutils.mnistloader"] = "nnutils/mnistloader.lua",
      ["nnutils.onehot"] = "nnutils/onehot.lua",
      ["nnutils.processingchain"] = "nnutils/processingchain.lua",
      ["nnutils.processor"] = "nnutils/processor.lua",
      ["nnutils.randomcrop"] = "nnutils/randomcrop.lua",
      ["nnutils.randomflip"] = "nnutils/randomflip.lua",
      ["nnutils.trainer"] = "nnutils/trainer.lua",
   }
}
