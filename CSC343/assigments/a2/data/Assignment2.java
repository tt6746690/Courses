import java.sql.*;
import java.util.List;

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
            return true;
        } catch (SQLException e) {
            System.err.println("SQL Exception. <Message>: " + se.getMessage());
            return false;
        }
    }

    @Override
    public boolean disconnectDB() {
        try {
            connection.close();
            return true;
        } catch (SQLException e) {
            System.err.println("SQL Exception. <Message>: " + se.getMessage());
            return false;
        }
    }

    @Override
    public ElectionCabinetResult electionSequence(String countryName) {

        String query; PreparedStatement ps; ResultSet rs;

        query = "select e.id as election_id, cab.id as cabinet_id " + 
                "from election e join country c on e.country_id=c.id" +
                                 "join cabinet cab on e.id=cab.election_id" + 
                "where c.name = ?" + 
                "order by e.e_date desc;";

        ps = connection.prepareStatement(query);
        ps.setString(1, countryName);
        rs = ps.executeQuery();

        List<Integer> elections = new ArrayList<Interger>();
        List<Integer> cabinets  = new ArrayList<Interger>();        
        while (rs.next()) {
            int election_id = rs.getInt("election_id");
            int cabinet_id = rs.getInt("cabinet_id");
            elections.add(election_id);
            cabinets.add(cabinet_id);
        }

        return new ElectionCabinetResult(elections, cabinets);
    }

    @Override
    public List<Integer> findSimilarPoliticians(Integer politicianName, Float threshold) {
        // Implement this method!
        return null;
    }

    public static void main(String[] args) {
        // You can put testing code in here. It will not affect our autotester.

        url = "jdbc:postgresql://localhost:5432/csc343h-wangpeiq";
        username = "wangpeiq";
        password = "";

        Assignment2 a2 = new Assignment2();


        ElectionCabinetResult ecResult = a2.electionSequence("Canada");
        assert ecResult.elections.size() == ecResult.cabinets.size();

        for(int i = 0; i < ecResult.elections.size(); i++) {
            System.out.println("election_id= " + ecResult.elections.get(i).toString() + 
                               " cabinets_id= " + ecResult.cabinets.get(i).toString());
        }
    }

}

