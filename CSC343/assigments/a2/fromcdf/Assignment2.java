import java.sql.*;
import java.util.List;
import java.util.ArrayList;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.Reader;

// If you are looking for Java data structures, these are highly useful.
// Remember that an important part of your mark is for doing as much in SQL (not Java) as you can.
// Solutions that use only or mostly Java will not receive a high mark.
//import java.util.ArrayList;
//import java.util.Map;
//import java.util.HashMap;
//import java.util.Set;
//import java.util.HashSet;
public class Assignment2 extends JDBCSubmission {

    public Assignment2() throws ClassNotFoundException {
        //  locate, load, and link the class or interface at runtime
        Class.forName("org.postgresql.Driver");
    }

    @Override
    public boolean connectDB(String url, String username, String password) {
        try {
            connection = DriverManager.getConnection(url, username, password);
            System.out.println("Successfully conected ...");
            return true;
        } catch (SQLException e) {
            System.err.println("SQL Exception. <Message>: " + e.getMessage());
            return false;
        }
    }

    @Override
    public boolean disconnectDB() {
        try {
            connection.close();
            System.out.println("Successfully closed connection ...");
            return true;
        } catch (SQLException e) {
            System.err.println("SQL Exception. <Message>: " + e.getMessage());
            return false;
        }
    }

    @Override
    public ElectionCabinetResult electionSequence(String countryName) {

        try {
            String query; PreparedStatement ps; ResultSet rs;

            query = "select e.id as election_id, cab.id as cabinet_id " +
                    "from parlgov.election e join country c on e.country_id=c.id " +
                                    "join cabinet cab on e.id=cab.election_id " +
                    "where c.name = ? " +
                    "order by e.e_date desc;";

            ps = connection.prepareStatement(query);
            ps.setString(1, countryName);
            rs = ps.executeQuery();

            List<Integer> elections = new ArrayList<Integer>();
            List<Integer> cabinets  = new ArrayList<Integer>();        

            while (rs.next()) {
                int election_id = rs.getInt("election_id");
                int cabinet_id = rs.getInt("cabinet_id");
                elections.add(election_id);
                cabinets.add(cabinet_id);
            }
            return new ElectionCabinetResult(elections, cabinets);
        } catch(SQLException e) {
            System.err.println("SQL Exception. <Message>: " + e.getMessage());
            return null;
        }
    }

    @Override
    public List<Integer> findSimilarPoliticians(Integer politicianName, Float threshold) {

        try {
            List<Integer> similarPresidents = new ArrayList<Integer>();

            String queryString1 = "select description, comment " + 
                                  "from politician_president " + 
                                  "where id = ?";
            PreparedStatement ps1 = connection.prepareStatement(queryString1);
            ps1.setInt(1, politicianName);
            ResultSet rs1 = ps1.executeQuery();

            String presidentDescription = "";
            String presidentComment = "";

            while (rs1.next()) {
                presidentDescription = rs1.getString("description");
                presidentComment = rs1.getString("comment");
            }

            String queryString2 = "select id, description, comment " + 
                                  "from politician_president " + 
                                  "where id != ?";
            PreparedStatement ps2 = connection.prepareStatement(queryString2);
            ps2.setInt(1, politicianName);
            ResultSet rs2 = ps2.executeQuery();

            while (rs2.next()) {
                Integer otherPresidentID = rs2.getInt("id");
                String otherPresidentDescription = rs2.getString("description");
                String otherPresidentComment = rs2.getString("comment");
                if(similarity(presidentDescription + " " + presidentComment, otherPresidentDescription + " " + otherPresidentComment) > threshold){
                    similarPresidents.add(otherPresidentID);
                }
            }
            return similarPresidents;

        } catch(SQLException e) {
            System.err.println("SQL Exception. <Message>: " + e.getMessage());
            return null;
        }
    }

    public static void main(String[] args) {
        // You can put testing code in here. It will not affect our autotester.

        String url = "jdbc:postgresql://localhost:5432/csc343h-wangpeiq?currentSchema=parlgov";
        String username = "wangpeiq";
        String password = "";
        String countryName = "Canada";

        try {
            Assignment2 a2 = new Assignment2();
            a2.connectDB(url, username, password);

            ElectionCabinetResult ecresult = a2.electionSequence(countryName);
            assert ecresult.elections.size() == ecresult.cabinets.size();

            System.out.println("electionSequence: ");
            for(int i = 0; i < ecresult.elections.size(); i++) {
                System.out.println("election_id= " + ecresult.elections.get(i).toString() +
                                   " cabinets_id= " + ecresult.cabinets.get(i).toString());
            }

            Integer politicianId = 9;
            Float threshold = new Float(-1);
            List<Integer> pres = a2.findSimilarPoliticians(politicianId, threshold);

            for(Integer i : pres) {
                System.out.println("politician id = " + i);
            }
            

        } catch(ClassNotFoundException e) {
            return;
        } 

    }

}

