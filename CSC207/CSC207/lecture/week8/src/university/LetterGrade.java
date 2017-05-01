package university;

/** A representation of a letter grade: A, B, C, D, or F. */
public class LetterGrade extends Grade {
  
    /** The letter value. */
    private String grade;

    /**
     * Creates a new LetterGrade with value grade.
     * @param grade the letter grade value (must be one of Grade.VALID_GRADES)
     */
    public LetterGrade(String grade) {
        this.grade = grade;
    }

    /*
     * This method is required because this class is not declared abstract 
     * and it inherits merely an abstract gpa method from Grade.
     */
    @Override
    public double gpa() {
        double gpa_value = 0.0;

        /*
         * Look up the details of the switch statement in the
         * Java documentation.
         */
        switch (this.grade) {
        case "A": gpa_value = 4.0; break;
        case "B": gpa_value = 3.0; break;
        case "C": gpa_value = 2.0; break;
        case "D": gpa_value = 1.0; break;
        case "F": gpa_value = 0.0;
        }

        return gpa_value;
    }

    @Override
    public String toString() {
        return "Letter grade " + grade;
    }
}

