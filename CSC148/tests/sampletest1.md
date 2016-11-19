### question 1

```python

class PollQuestion:
  '''
  the pollquestion class records a single question on an opinion poll and the responses they have been received

  === attributes ===
  @param str question: the poll question
  @param list respondents: email of people who responded with a valid response
  @param dict[str:int] poll: a poll tallying the number of yes, no, and maybe

  '''
  def __init__(self, question):
    '''
    initialize the class

    @param PollQuestion self: information regarding the poll question
    @param str question: the poll question
    @rtype: None
    '''
    self.question = question
    self.respondents = []
    self.poll = {'yes': 0, 'no': 0, 'maybe': 0}


  def record(self, res, email):
    '''
    recording a valid response to the poll question
    ignore poll if they have responded. The legal response is yes no and maybe

    @param PollQuestion self: this Poll question
    @param str res: response to the poll question
    @param str email: email of the respondents
    @rtype: None
    '''
    if (res in ['yes', 'no', 'maybe'] and email not in self.respondents):
        self.respondents.append(email)
        self.poll[res] += 1

  def __eq__(self, other):
    '''
    return a boolean whether if self is equivalent to other

    @param PollQuestion self: this Poll question
    @param PollQuestion other: other Poll question
    @rtype: bool
    '''

    return (isinstance(other, PollQuestion) and
            self.question == other.question and
            self.respondents == other.respondents and
            self.poll == other.poll)
```


### question 3

```python
class Student:
'''A student with these attributes:
name: str -- the name of the student
student_number: int -- student number
num_credits: int -- the number of credits the student has earned.
This is an abstract class. Only a child class should be instantiated.
'''
  def __init__(self, name, student_number):
  '''
  (Student, str, int) -> NoneType
  Initialize this student (self) with name, student_number, and
  zero credits so far.
  '''
    self.name, self.student_number = name, student_number
    self.num_credits = 0

  def complete_course(self, grade):
  '''
  (Student, int) -> NoneType
  This student (self) has completed a course with this grade.
  Update this student (self) if grade is high enough.
  '''
    raise(NotImplementedError)


  def can_graduate(self):
  '''
  (Student) -> bool
  Return whether or not this student (self) satisfies the graduation
  requirements.
  '''
    raise(NotImplementedError)



class UndergradStudent(Student):
''' An undergraduate student with these additional attributes:
major: str -- the student's major
'''

  def __init__(self, name, student_number, major):
  ''' (UndergradStudent, str, int, str) -> NoneType
  Initialize this undergrad student (self) with name, student_number,
  zero credits so far, and major.
  '''
    Student.__init__(self, name, student_number)
    self.major = major


  def complete_course(self, grade):
  ''' (UndergradStudent, int) -> NoneType
  This student (self) has completed a course with this grade.
  Update this student (self) if grade is high enough.
  >>> ug = UndergradStudent('Fred', 1, 'paleontology')
  >>> ug.complete_course(86)
  >>> ug.num_credits
  1
  >>> ug.complete_course(30)
  >>> ug.num_credits
  1
  '''
    if grade >= 50:
      self.num_credits += 1


  def can_graduate(self):
  ''' (UndergradStudent) -> bool
  Return whether or not this undergrad student (self) satisfies
  the graduation requirements.
  >>> ug = UndergradStudent('Fred', 1, 'paleontology')
  >>> ug.can_graduate()
  False
  >>> ug.num_credits = 20
  >>> ug.can_graduate()
  True
  '''
    return self.num_credits >= 20


  if __name__ == '__main__':
    import doctest
    doctest.testmod()

```
