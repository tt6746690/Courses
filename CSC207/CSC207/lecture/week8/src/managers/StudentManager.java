package managers;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInput;
import java.io.ObjectInputStream;
import java.io.ObjectOutput;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Scanner;
import java.util.logging.Handler;
import java.util.logging.Level;
import java.util.logging.ConsoleHandler;
import java.util.logging.Logger;

import university.Student;

/**
 * Manages the saving and loading of Students.
 */
public class StudentManager {

    // Added Week 8, Lecture 1:
    // All instances of StudentManager will share one Logger, so it's
    // static.
    // We use StudentManager.class.getName() to get the fully-qualified
    // name of this class, which happens to be "managers.StudentManager".
    // We use this name as the name of our Logger object.
    private static final Logger logger =
            Logger.getLogger(StudentManager.class.getName());
    private static final Handler consoleHandler = new ConsoleHandler();
    
    /** A mapping of student ids to Students. */
    private Map<String, Student> students;
    
    // Revised Week 8, Lecture 1:
    /**
     * Creates a new empty StudentManager.
     * @throws IOException 
     * @throws ClassNotFoundException 
     */
    public StudentManager(String filePath) throws ClassNotFoundException, IOException {

        students = new HashMap<String, Student>();
        
        // Associate the handler with the logger.
        logger.setLevel(Level.ALL);
        consoleHandler.setLevel(Level.ALL);
        logger.addHandler(consoleHandler);
        
        // Reads serializable objects from file.
        // Populates the record list using stored data, if it exists.
        File file = new File(filePath);
        if (file.exists()) {
            readFromFile(filePath);
        } else {
            file.createNewFile();
        }
    }
    
    /**
     * Populates the records map from the file at path filePath.
     * @param filePath the path of the data file  
     * @throws FileNotFoundException if filePath is not a valid path
     */
    public void readFromCSVFile(String filePath) 
            throws FileNotFoundException {
        
        // FileInputStream can be used for reading raw bytes, like an image. 
        Scanner scanner = new Scanner(new FileInputStream(filePath));
        String[] record;
        Student student;

        while(scanner.hasNextLine()) {
            record = scanner.nextLine().split(",");
            student = new Student(record[0].split(" "), 
                    record[1], record[2], record[3]);
            students.put(student.getID(), student);
        }
        scanner.close();
    }
    
    // Added Week 8, Lecture 1:
    private void readFromFile(String path) throws ClassNotFoundException {

        try {
            InputStream file = new FileInputStream(path);
            InputStream buffer = new BufferedInputStream(file);
            ObjectInput input = new ObjectInputStream(buffer);

            //deserialize the Map
            students = (Map<String,Student>) input.readObject();
            input.close();
        } catch (IOException ex) {
            logger.log(Level.SEVERE, "Cannot read from input.", ex);
        }    
    }
    
    /**
     * Adds record to this StudentManager.
     * @param record a record to be added.
     */
    public void add(Student record) {
        students.put(record.getID(), record);
        
        // Added Week 8, Lecture 1:
        // Log the addition of a student.
        logger.log(Level.FINE, "Added a new student " + record.toString());
    }
    
    /**
     * Writes the students to file at filePath.
     * @param filePath the file to write the records to
     * @throws IOException 
     */
    public void saveToFile(String filePath) throws IOException {

        OutputStream file = new FileOutputStream(filePath);
        OutputStream buffer = new BufferedOutputStream(file);
        ObjectOutput output = new ObjectOutputStream(buffer);

        // serialize the Map
        output.writeObject(students);
        output.close();
    }
    
    @Override
    public String toString() {
        String result = "";
        for (Student r : students.values()) {
            result += r.toString() + "\n";
        }
        return result;
    }
}

