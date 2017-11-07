// A simple JDBC example.

// Remember that you need to put the jdbc postgresql driver in your class path
// when you run this code.
// See /local/packages/jdbc-postgresql on cdf for the driver, another example
// program, and a how-to file.

// To compile and run this program on cdf:
// (1) Compile the code in Example.java.
//     javac Example
// This creates the file Example.class.
// (2) Run the code in Example.class.
// Normally, you would run a Java program whose main method is in a class 
// called Example as follows:
//     java Example
// But we need to also give the class path to where JDBC is, so we type:
//     java -cp /local/packages/jdbc-postgresql/postgresql-9.4.1212.jar: Example
// Alternatively, we can set our CLASSPATH variable in linux.  (See
// /local/packages/jdbc-postgresql/HelloPostgresql.txt on cdf for how.)

import java.sql.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;

class JDBCJellyBean {
    
    public static void main(String args[]) throws IOException
        {
            String url;
            Connection conn;
            PreparedStatement pStatement;
            ResultSet rs;
            String queryString;

            try {
                Class.forName("org.postgresql.Driver");
            }
            catch (ClassNotFoundException e) {
                System.out.println("Failed to find the JDBC driver");
            }
            try
            {
                // This program connects to my database csc343h-dianeh,
                // where I have loaded a table called Guess, with this schema:
                //     Guesses(_number_, name, guess, age)
                // and put some data into it.
                
                // Establish our own connection to the database.
                // This is the right url, username and password for jdbc
                // with postgres on cdf -- but you would replace "dianeh"
                // with your cdf account name.
                // Password really does need to be the emtpy string.
                url = "jdbc:postgresql://localhost:5432/csc343h-wangpeiq";
                conn = DriverManager.getConnection(url, "wangpeiq", "");

                // The next query depends on user input, so we are wise to
                // prepare it before inserting the user input.
                queryString = "select avg(guess) as avgguess from guesses where age >= ?";
                PreparedStatement ps = conn.prepareStatement(queryString);

                // Find out what string to use when looking up guesses.
                BufferedReader br = new BufferedReader(new 
                      InputStreamReader(System.in));
                System.out.println("Age at least? ");
                String line = br.readLine();
                int ageAtLeast = Integer.parseInt(line);

                // Insert that string into the PreparedStatement and execute it.
                ps.setInt(1, ageAtLeast);
                rs = ps.executeQuery();

                // Iterate through the result set and report on each tuple.
                while (rs.next()) {
                    int avgGuess = rs.getInt("avgguess");
                    System.out.println("average guess amoung people at least " + line + " is " + avgGuess);
                }

                // Create an array to put names into array
                queryString = "select distinct name from guesses";
                pStatement = conn.prepareStatement(queryString);
                rs = pStatement.executeQuery();

                Collection<String> collection = new ArrayList<String>();
                while(rs.next()){
                    String name = rs.getString("name");
                    collection.add(name);
                }

                for (Iterator<String> iterator = collection.iterator(); iterator.hasNext();) {
                    System.out.println("value= " + iterator.next());
                }


                
            }
            catch (SQLException se)
            {
                System.err.println("SQL Exception." +
                        "<Message>: " + se.getMessage());
            }

        }
        
}
