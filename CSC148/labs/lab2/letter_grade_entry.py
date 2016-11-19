
from grade import GradeEntry
class LetterGradeEntry(GradeEntry):
    '''
    a grade entry with letter grade

    '''

    def __init__(self, id, grade, weight):
        '''
        creates a letter grade entry inheriting from grade entry

        overrides GradeEntry.__init__

        @param LetterGradeEntry self: the letter grade entry object
        @param id str: course identifier
        @param grade str | int: course grade
        @param weight: course weighting
        @rtype: None

        >>> letter1 = LetterGradeEntry("his450", "B+", 1.0)
        >>> letter1.id
        'his450'
        >>> letter1.grade
        'B+'
        >>> letter1.weight
        1.0
        '''
        GradeEntry.__init__(self, id, grade, weight)
        self._set_point

    def _set_point(self):
        '''
        generate letter grade entry's grade point from letter grade

        overrides GradeEntry._set_point   # need to state this


        @type self: LetterGradeEntry
        @rtype: None

        >>> letter2 = LetterGradeEntry("his450", "B+", 1.0)
        >>> letter2.point
        3.3
        '''
        letterDict = {
            "A+": 4.0,
            "A": 4.0,
            "A-": 3.7,
            "B+": 3.3,
            "B": 3.0,
            "B-": 2.7,
            "C+": 2.3,
            "C": 2.0,
            "C-": 1.7,
            "D+": 1.3,
            "D": 1.0,
            "D-": 0.7,
            "F": 0
        }
        self._point = letterDict[self.grade]


if __name__ == '__main__':
    letter2 = LetterGradeEntry("his450", "B+", 1.0)
    print(letter2.point)
    import doctest
    doctest.testmod()
