package photo_renamer;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Date;

/** An object to keep track of the full logs of one photo
 *
 */

public class LogEntry implements Serializable {

	/**serialVersionUID so that LogEntry is serializable */
	private static final long serialVersionUID = 1L;
	/** Current name of the photo */
	private String currName;
	/** All the changes made to the photo */
	private ArrayList<String[]> entries;

	/** Constructor
	 * 
	 * @param currName
	 */
	public LogEntry(String currName) {
		this.setCurrName(currName);
		this.entries = new ArrayList<String[]>();
		this.newName(currName);
	}

	/**
	 * Logs a new name change.
	 * 
	 * @param newName
	 */
	public void newName(String newName) {
		this.setCurrName(newName);
		String[] entry = new String[2];
		entry[0] = newName;
		Date date = new Date();
		entry[1] = date.toString();
		entries.add(entry);
	}

	/** 
	 * @param index representing the state of name the user wants to revert back to
	 * @return the name the photo should be reverted to based on an index inputted by the user
	 */
	public String getRevertedName(int index) {
		String revertedName = entries.get(index - 1)[0];
		return revertedName;
	}

	/*
	public String toString() {
		String output = new String();
		output += this.currName + "\n";
		for (int i = 0; i < this.entries.size(); i++) {
			output += (i + 1) + " " + entries.get(i)[0] + " " + entries.get(i)[1];
			output += "\n";
		}
		return output;
	}*/
	
	/** toString method	
	 * @return a string representation of a LogEntry
	 */
	public String toString() {
		String output = new String();
		output += "<html>" + this.getCurrName() + "<br>";
		for (int i = 0; i < this.entries.size(); i++) {
			output += (i + 1) + " " + entries.get(i)[0] + " " + entries.get(i)[1];
			output += "<br>";
		}
		output += "</html>";
		return output;
	}

	/** Getter method for the current name of the photo		
	 * 
	 * @return current name of the photo
	 */
	String getCurrName() {
		return currName;
	}

	/** Setter for the current name of the photo
	 * 
	 * @param new name for the photo
	 */
	void setCurrName(String newName) {
		this.currName = newName;
	}
	
	/** Main method for testing purposes.
	 * 
	 * @param args
	 * @throws InterruptedException
	 */
	public static void main(String[] args) throws InterruptedException {
		LogEntry e = new LogEntry("Name1");
		e.newName("Name2");
		e.newName("Name3");
		System.out.println(e.toString());
	}
}