package university;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

// 2 Oct: We began working on this new subclass.

// Week 4, lecture 2: We declared that Student implements the IDed 
// interface. That required us to define a method getID, as specified 
// by the interface.

//Week 7, Lecture 2: Added "implements Serializable".
public class Student extends Person implements IDed<String>, Serializable {
    
    /**
     * 
     */
    private static final long serialVersionUID = 3950536640070526829L;
    
    private String studentNum;
	// 2 October: I added this after class, and made it public in order
	// to demonstrate "shadowing".
	public String motto;
	private static int studentCount;
	// Week 5, lecture 1: Added this instance variable to record
	// this student's grades.
	private Map<String, Grade> courseToGrade;

	// We generated this constructor automatically,
	// and saw that it used super to call the appropriate
	// constructor from the parent class. This avoids
	// duplicate code.
	public Student(String[] name, String dob, String gender, String studentNum) {
		super(name, dob, gender);
		// With dob a private attribute in Person,
		// even class Student, of which every instance HAS a dob,
		// cannot touch it.
		// With protected level access, this is allowed.
		// But it's stupid so let's not do it.
//		this.dob = "never";
		this.studentNum = studentNum;
		// 2 October: I added this after class.
		this.motto = "Free pizza!!";
		studentCount += 1;
		// Week 5, lecture 1: Added this to construct an empty HashMap
		// that can hold this student's grades.
		this.courseToGrade = new HashMap<String, Grade>();
	}
	
	// Week 5, lecture 1: Wrote this method for recording a Grade.
	// It is capable of taking any subclass of Grade, including a
	// LetterGrade, a NumericGrade, or even some future subclass we
	// might define, such as CreditNoCreditGrade.
	public void addGrade(String course, Grade g) {
	    this.courseToGrade.put(course, g);
	}
	
	// Week 6, lecture 2: Added these alternative forms of addGrade,
	// which could be preferred by client code, since they don't require
	// the client to construct a Grade object.
	public void addGrade(String course, int g) throws InvalidGradeException {
	    // Whenever we construct a NumericGrade, an InvalidGradeException
	    // could be thrown.  Because InvalidGradeException extends Exception
	    // and not RuntimeException, it is a "checked" exception.  Therefore,
	    // We must either catch it, or declare that we throw it.
	    this.courseToGrade.put(course, new NumericGrade(g));
	}
	
	public void addGrade(String course, String g) {
	    // Why don't we have to handle an exception when we construct
	    // a LetterGrade?
	    this.courseToGrade.put(course, new LetterGrade(g));
	}
	
	// Week 5, lecture 1: Added after class.
	// What if we wanted client code to deal with Strings and ints,
	// and never have to construct Grade objects.  Instead of the 
	// above method, we would need two different addGrade methods:
	// one that takes a String and one that takes an int.
	// Exercise: write those two methods.
	
	// Another reason to have two different addGrade methods would
	// be if there are specific actions we want to take for one type
	// of grade but not the other (in addition to recording the grade
	// in this.courseToGrade).
	
	// Any time we have more than one method with the same name,
	// we say that that method is "overloaded".  Even though they
	// have the same name, Java can tell the methods apart, because
	// they have different signatures.
	
	// You have seen overloaded methods before, when we saw multiple
	// constructors in a single class.
	
	// Overloaded is different from the concept of overridden!  To help
	// remember which is which, think of the expression in English:
	// "That's a loaded question", which implies that the question 
	// is loaded with meaning.  In Java, a loaded method is a
	// method name that is loaded with meaning: it could mean
	// one method body, or it could mean another.
	
	// For more on overloading, see:
	// https://docs.oracle.com/javase/tutorial/java/javaOO/methods.html
	
	public static int getStudentCount() {
		return studentCount;
	}

	// We decided not to have a setter for this attribute,
	// since a student's student number should not change over time.
	public String getStudentNum() {
		return studentNum;
	}

	// We overrode the toString from class Person.
	// Again, we avoided duplicating code by calling the
	// method in the parent class to do the parent-related
	// parts of the job.
	@Override
	public String toString() {
		return super.toString() + " , " + this.studentNum;
	}

	public String getID() {
	    return this.studentNum;
	}
}
