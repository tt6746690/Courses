class GradeEntry:
    '''
    a Grade entry class that gets and sets
    1. course identifier
    2. coures weight
    3. course grade (specified in subclass)
    and also returns the grade point.

    === Attributes ===
    @param id str: a course identifier
    @param weight float: the weight of a course
    @param grade str | float: the grade of a course
    @param point float: the grate point of a course

    '''

    def __init__(self, id, grade, weight):
        '''
        create a new Grade entry self defined by id, weight, and grade

        @param GradeEntry self: this GradeEntry object
        @param id str: course identifier
        @param grade str | int: course grade
        @param weight: course weighting
        @rtype: None
        '''
        self._set_id(id)
        self._set_grade(grade)
        self._set_weight(weight)
        self._set_point()

    # when to use getter and setters python. not so important in python. use it on private variable mostly

    def _get_id(self):
        '''
        returns the id of the grade entry
        although not necessary since id is a public property that is specified in the constructor

        @type self: GradeEntry
        @rtype: str
        '''
        return self._id

    def _set_id(self, value):
        '''
        sets the id of the grade entry

        @type self: GradeEntry
        @type value: course identifier
        @rtype: None
        '''
        self._id = value

    def __eq__(self, other):
        return (type(self) == type(other) and
        self.grade == other.grade and
        self.id == other.id and
        self.weight == other.weight)

    id = property(_get_id, _set_id)

    def _get_grade(self):
        return self._grade                  # to make people not altering public variables -> use getters and setters
    def _set_grade(self, value):
        self._grade = value
    grade = property(_get_grade, _set_grade)

    def _get_weight(self):
        return self._weight
    def _set_weight(self, value):
        self._weight = value
    weight = property(_get_weight, _set_weight)


    def _set_point(self, value):
        raise NotImplementedError("grade points are generated in subclasses")
    def _get_point(self):
        return self._point
    point = property(_get_point, _set_point)
