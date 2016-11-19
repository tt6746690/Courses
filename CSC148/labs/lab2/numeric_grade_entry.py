
from grade import GradeEntry
class NumericGradeEntry(GradeEntry):
    '''
    a grade entry with numeric grade

    '''

    def __init__(self, id, grade, weight):
        '''
        creates a numeric grade entry inheriting from grade entry

        overrides GradeEntry.__init__

        @param NumericGradeEntry self: the letter grade entry object
        @param id str: course identifier
        @param grade str | int: course grade
        @param weight: course weighting
        @rtype: None

        >>> Numeric1 = NumericGradeEntry("csc148", 87, 1.0)
        >>> Numeric1.id
        'csc148'
        >>> Numeric1.point
        4.0
        >>> Numeric1.weight
        1.0
        '''
        GradeEntry.__init__(self, id, grade, weight)
        self._set_point

    def _set_point(self):
        '''
        generate numeric grade entry's grade point from numeric grade

        overrides GradeEntry._set_point   # need to state this


        @type self: NumericGradeEntry
        @rtype: None

        >>> Numeric1 = NumericGradeEntry("csc148", 87, 1.0)
        >>> Numeric1.point
        4.0
        '''
        if (85 <= self.grade <= 100):
            self._point = 4.0
        if (80 <= self.grade <= 84):
            self._point = 3.7
        if (77 <= self.grade <= 79):
            self._point = 3.3
        if (73 <= self.grade <= 76):
            self._point = 3.0
        if (70 <= self.grade <= 72):
            self._point = 2.7
        if (67 <= self.grade <= 69):
            self._point = 2.3
        if (63 <= self.grade <= 66):
            self._point = 2.0
        if (60 <= self.grade <= 62):
            self._point = 1.7
        if (57 <= self.grade <= 59):
            self._point = 1.3
        if (53 <= self.grade <= 56):
            self._point = 1.0
        if (50 <= self.grade <= 52):
            self._point = 1.0
        if (0 <= self.grade <= 49):
            self._point = 0.0


if __name__ == '__main__':
    import doctest
    doctest.testmod()
