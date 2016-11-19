from numeric_grade_entry import NumericGradeEntry
from letter_grade_entry import LetterGradeEntry

if __name__ == "__main__":
    grades = [NumericGradeEntry("csc148", 87, 1.0),
        NumericGradeEntry("bio150", 76, 2.0),
        LetterGradeEntry("his450", "B+", 1.0)]
    for g in grades:
        print("Weight: {}, grade: {}, points: {}".format(g.weight, g.grade, g.point))

    # Use methods or attributes of g to compute weight times points
    total = sum([ g.weight * g.point for g in grades ])

    # sum up the credits
    total_weight = sum([g.weight for g in grades])

    print("GPA = {}".format(total / total_weight))
