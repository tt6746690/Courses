// Aside: some useful Eclipse things we used:
// Help -> search box
// Source -> Toggle Comment
// Source -> Format

package university;

import java.io.Serializable;
import java.util.Arrays;

// Week 7, Lecture 2: Added "implements Serializable".
public class Person implements Serializable {

	// Week 8, Lecture 1: Used Eclipse to generate this.
    private static final long serialVersionUID = -9022720262689191328L;
    
    // Week 8, Lecture 2: Used Eclipse to generate these two methods,
    // hashCode and equals.
    // If you override one, you should override the other, and if you
    // implement Serializable, you should override both.
    // The reasons are subtle, and beyond the scope of csc207, but
    // here is a useful reference:
    // http://www.ibm.com/developerworks/library/j-jtp05273/
    @Override
    public int hashCode() {
        final int prime = 31;
        int result = 1;
        result = prime * result + ((dob == null) ? 0 : dob.hashCode());
        result = prime * result + Arrays.hashCode(name);
        return result;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj)
            return true;
        if (obj == null)
            return false;
        if (getClass() != obj.getClass())
            return false;
        Person other = (Person) obj;
        if (dob == null) {
            if (other.dob != null)
                return false;
        } else if (!dob.equals(other.dob))
            return false;
        if (!Arrays.equals(name, other.name))
            return false;
        return true;
    }

    protected String[] name;
	protected String dob;
	protected String gender;
	
	// 2 October: I added this after class, and made it public in order
	// to demonstrate "shadowing".
	/** This Person's motto. */
	public String motto;

	// We generated these constructors automatically using
	// Source -> Generate Constructor Using Fields
	
	/**
	 * Creates a new Person with name name, date of birth dob, and
	 * gender gender.
	 * 
	 * @param name the name of this Person.
	 * @param dob the date of birth of this Person, in DDMMYY format.
	 * @param gender the gender of this Person.
	 */
	public Person(String[] name, String dob, String gender) {
		// super(); We'll explain this later.
		this.name = name;
		this.dob = dob;
		this.gender = gender;
		// 2 October: I added this after class.
		this.motto = "Live long and prosper";
	}

	public Person(String[] name) {
		super();
		this.name = name;
	}

	// We made this constructor by typing it.
	// public Person(String[] name, String dob, String gender) {
	// this.name = name;
	// this.dob = dob;
	// this.gender = gender;
	// }

	// 2 Oct: We wrote this getter/setter pair by hand.
	public String getGender() {
		return this.gender;
	}

	public void setGender(String gender) {
		this.gender = gender;
	}

	// 2 Oct: We generated these getter/setter pairs automatically using
	// Source -> Generate Getters and Setters
	public String[] getName() {
		return name;
	}

	public void setName(String[] name) {
		this.name = name;
	}

	public String getDob() {
		return dob;
	}

	public void setDob(String dob) {
		this.dob = dob;
	}

	// 2 Oct: We overrode the toString inherited from Object.
	// Adding the @Override annotation allows Eclipse to warn
	// us if we think we are overriding, but we actually aren't,
	// e.g., due to a typo in the method name. This can save us
	// from some nasty bugs.
	@Override
	public String toString() {
		String result = new String("");
		for (String n : this.name) {
			result = result + n + " ";
		}
		result += this.dob;
		// Note the syntax of an if-statement.
		// For more, see:
		// http://docs.oracle.com/javase/tutorial/java/nutsandbolts/if.html
		if (this.gender.equals("M")) {
			result += ", male";
		} else {
			result += ", female";
		}
		return result;
	}
}
