package university;

import java.io.File;
import java.io.IOException;

import managers.StudentManager;

public class DemoManager {

    public static void main(String[] args) 
            throws IOException, ClassNotFoundException {
        demoStudentManager();
    }

    public static void demoStudentManager() 
            throws IOException, ClassNotFoundException {
    	
//        String csvPath =
//                "/Users/pgries/Documents/dept/courses/207/course-20716f/website/lectures/L0301/w8/src/managers_logging/students.csv";
        // Week 8 Lecture 1: Replaced the CSV code below with this:
        String path =
                "/Users/pgries/Documents/dept/courses/207/course-20716f/website/lectures/L0301/w8/src/managers_logging/students.ser";
        StudentManager sm = new StudentManager(path); 
        System.out.println(sm);

//        // Loads data from a CSV for first launch of the program
//        sm.readFromCSVFile(csvPath);
//        System.out.println(sm);
                
        // adds a new student to StudentManager sm's records
        sm.add(new Student(new String[] {"New", "Student"},
                "10102000", "F", "1122334455"));
        System.out.println(sm);
                
        // Writes the existing Student objects to file.
        // This data is serialized and written to file as a sequence of bytes.
        sm.saveToFile(path);
    }
}