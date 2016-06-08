local StudentTeacher, parent = torch.class('nn.StudentTeacher', 'nn.ConcatTable')

StudentTeacher._add = StudentTeacher.add
StudentTeacher.add = nil

function StudentTeacher:__init(studentNet, teacherNet)
  parent.__init(self)
  local teacher = nn.SwithableContainer(false)
  teacher:add(teacherNet)

  -- add teacher and student to self
  self:_add(teacher)
  self:_add(studentNet)
end

