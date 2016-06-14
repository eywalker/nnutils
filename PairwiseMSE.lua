local PairwiseMSE, parent = torch.class('nn.PairwiseMSE', 'nn.Sequential')

function PairwiseMSE:__init(...)
  parent.__init(self, ...)
  self:add(nn.CSubTable())
  self:add(nn.Power(2))
  self:add(nn.Mean(1, 1))
end

