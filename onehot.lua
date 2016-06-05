nnutils = nnutils or {}

function nnutils.onehot(v, N)
  local oh = torch.zeros(v:size(1), N)
  local vv = v:type('torch.LongTensor'):view(-1, 1)
  oh:scatter(2, vv, 1)
  return oh
end
