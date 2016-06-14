local StudentTeacher, parent = torch.class('nn.StudentTeacher', 'nn.ConcatTable')

-- rename add method to discourage usage outside of __init
StudentTeacher._add = StudentTeacher.add
StudentTeacher.add = nil

function StudentTeacher:__init(studentNet, teacherNet)
  parent.__init(self)
  local teacher = nn.SwitchableContainer(false)
  teacher:add(teacherNet)

  -- add teacher and student to self
  self:add(studentNet)
  self:add(teacher)
end

function StudentTeacher:fillGradOutput(gradOutput)
  if #gradOutput == 1 then
    gradOutput[2] = self:get(2).output:clone():fill(0)
  end
end

function StudentTeacher:updateGradInput(input, gradOutput)
  self:fillGradOutput(gradOutput)
  return parent.updateGradInput(self, input, gradOutput)
end

function StudentTeacher:backward(input, gradOutput, scale)
  self:fillGradOutput(gradOutput)
  return parent.backward(self, input, gradOutput, scale)
end

function StudentTeacher:accGradParameters(input, gradOutput, scale)
  self:fillGradOutput(gradOutput)
  return parent.accGradParameters(self, input, gradOutput, scale)
end

function StudentTeacher:accUpdateGradParameters(input, gradOutput, lr)
  self:fillGradOutput(gradOutput)
  return parent.accUpdateGradParameters(self, input, gradOutput, lr)
end
