package university;

/** A representation of a numeric grade (0 to 100). */
public class NumericGrade extends Grade {

    /** A numeric grade value. */
    private int grade;

    /**
     * Creates a new NumericGrade with value of grade.
     * @param grade the value of the new NumericGrade, must be
     * in [0, 100]
     * @throws InvalidGradeException if grade is not in [0, 100].
     */
    public NumericGrade(int grade) throws InvalidGradeException {
        if (grade < 0 || grade > 100) {
            throw new InvalidGradeException("Invalid grade " + grade);
        }
        this.grade = grade;
    }

    /*
     * This method is required: it is abstract in Grade.
     */
    @Override
    public double gpa(){
        if (grade < 50) { return 0.0; }
        if (grade < 53) { return 0.7; }
        if (grade < 57) { return 1.0; }
        if (grade < 60) { return 1.3; }
        if (grade < 63) { return 1.7; }
        if (grade < 67) { return 2.0; }
        if (grade < 70) { return 2.3; }
        if (grade < 73) { return 2.7; }
        if (grade < 77) { return 3.0; }
        if (grade < 80) { return 3.3; }
        if (grade < 84) { return 3.7; }
        return 4.0;
    }

    @Override
    public String toString() {
        return "Numeric grade " + grade;
    }
}