--[[ A criterion class that simply "feeds forward" the input
--as the cost while ignoring (any) targets passed in. This seemingly
--useless class is useful for training a network in unsupervised fashion
--while using an otherwise supervised training frameworks such as `dp`.
--
--By default criterion will work to minimize the input. Pass in `false` to
--invert the sign on the returned gradient, thus effectively working to
--maximize the input into this unit.
--]]

local FFCriterion, parent = torch.class('nn.FFCriterion', 'nn.Criterion')

function FFCriterion:__init(minimize)
    parent.__init(self)
    self.sign = (minimize==nil or minimize) and 1 or -1
    self.output = torch.Tensor()
    self.gradInput = torch.Tensor()
end

function FFCriterion:updateOutput(input, target)
    return input:mean()
end

function FFCriterion:updateGradInput(input, target)
    self.gradInput:resize(input:size()):fill(1):div(input:size(1)):mul(self.sign)
    return self.gradInput
end
