package university;

/*
 * An abstract method is one that is declared without an implementation.

 * An abstract class:
 * - cannot be instantiated, but can be subclassed
 * - it can, but is not required to, contain abstract methods
 *  
 * If a class contains an abstract method, then the class must be abstract.
 */

/**
 * A representation of a grade.
 */
public abstract class Grade implements Comparable<Grade> {

	/*
	 * VALID_GRADES is a constant. Because VALID_GRADES is declared "final",
	 * its value cannot be changed.
	 */
	/** Valid letter grades. */
	public static final String[] VALID_GRADES = { "A", "B", "C", "D", "F" };

	/*
	 * This method is abstract. It is a requirement for all subclasses of Grade
	 * to implement a method called gpa that takes no parameters and returns a
	 * double. Try removing gpa() from one of the subclasses and see what
	 * happens.
	 */
	/**
	 * Return the GPA that corresponds to this Grade.
	 * 
	 * @return the GPA that corresponds to this Grade.
	 */
	public abstract double gpa();

	/*
	 * Another example of a static method.
	 */
	/**
	 * Return a letter corresponding to the integer grade.
	 * 
	 * @param grade
	 *            the integer value of the grade, must be in [0, 100]
	 * @return the letter that corresponds to grade
	 */
	public static String toLetter(int grade) {
		if (grade < 50) {
			return "F";
		}
		if (grade < 60) {
			return "D";
		}
		if (grade < 70) {
			return "C";
		}
		if (grade < 80) {
			return "B";
		}
		return "A";
	}
	
	/**
	 * Compares this grade to another grade.
	 * 
     * @param  other The grade to compare this grade to
     * @return       A negative integer, zero, or a positive integer
     *               as this grade is less than, equal to, or greater 
     *               than grade other, according to their gpa() values.
	 */
    @Override
    public int compareTo(Grade other) {
        return (new Double(this.gpa()).compareTo(new Double(other.gpa())));
    }

}
