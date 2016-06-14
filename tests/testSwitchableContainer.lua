require 'nnutils'

teacher = nn.SwitchableContainer(false)

net1 = nn.Linear(10, 10)
net1.weight:fill(1)
net1.bias:fill(1)

teacher:add(net1)

input = torch.Tensor(10):random(1, 5)
err = torch.randn(10)

output = teacher:forward(input)
teacher:zeroGradParameters()
teacher:backward(input, err)

print(teacher:get(1).gradWeight)

