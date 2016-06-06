nnutils = nnutils or {}

function nnutils.topkScore(outputs, labels, N)
  N = N or 1 -- defaults to top 1
  local batchSize = outputs:size(1)
  local topk = 0
  local topk_values = outputs:topk(N, 2, true, true) -- descending and sorted
  for i=1,batchSize do
    if topk_values[i][N] <= outputs[i][labels[i]] then
      topk = topk + 1
    end
  end
  return topk
end
