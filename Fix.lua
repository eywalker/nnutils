nnutils = nnutils or {}

-- function that modifies the passed in module becomes "fixed"
-- i.e. it will no longer be trainable via standard calls like
-- updateParamters. Also any method that would cause it's gradient
-- with respect to parameters to change is disabled (e.g.
-- accGradParameters
-- currently this operation is NOT reversible

function nnutils.Fix(module)
  module.fixed = true
  module:zeroGradParameters()
  module.accGradParameters = function() end
end
