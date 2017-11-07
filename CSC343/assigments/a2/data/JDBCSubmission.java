import java.sql.Connection;
import java.util.List;
import java.util.Set;

public abstract class JDBCSubmission {

    /**
     * The return class for method electionSequence.
     */
    public class ElectionCabinetResult {
        public List<String> elections;
        public List<String> cabinets;
        public ElectionCabinetResult(List<String> elections, List<String> cabinets) {
            this.elections = elections;
            this.cabinets = cabinets;
        }
        @Override
        public boolean equals(Object obj) {
            if (!(obj instanceof ElectionCabinetResult)) {
                return false;
            }
            ElectionCabinetResult other = (ElectionCabinetResult) obj;
            return this.elections.equals(other.elections) &&
                   this.cabinets.equals(other.cabinets) &&
        }
    }

    /**
     * Computes the Jaccard similarity of two strings.
     *
     * The Jaccard similarity is a measure used for comparing the similarity of sets.
     * The Jaccard similarity of two strings is defined on the set of tokens of the strings.
     * It is the size of the intersection divided by the size of the union of the token sets.
     *
     * @param  a  the first string
     * @param  b  the second string
     * @return    the Jaccard similarity (between 0.0 and 1.0 inclusively) of a and b
     */
    protected double similarity(String a, String b) {
        a = a.replaceAll("[^a-zA-Z0-9]", " ").toLowerCase();
        b = b.replaceAll("[^a-zA-Z0-9]", " ").toLowerCase();
        final java.util.regex.Pattern p = java.util.regex.Pattern.compile("\\s+");
        Set<?> left = p.splitAsStream(a).collect(java.util.stream.Collectors.toSet());
        Set<?> right = p.splitAsStream(b).collect(java.util.stream.Collectors.toSet());
        final int sa = left.size();
        final int sb = right.size();
        if ((sa - 1 | sb - 1) < 0)
            return 0.0;
        if ((sa + 1 & sb + 1) < 0)
            return 0.0;
        final Set<?> smaller = sa <= sb ? left : right;
        final Set<?> larger  = sa <= sb ? right : left;
        int intersection = 0;
        for (final Object element : smaller) try {
            if (larger.contains(element))
                intersection++;
        } catch (final ClassCastException | NullPointerException e) {}
        final long sum = (sa + 1 > 0 ? sa : left.stream().count())
                + (sb + 1 > 0 ? sb : right.stream().count());
        return 1d / (sum - intersection) * intersection;
    }

    /**
     * The connection used for this session.
     */
    public Connection connection;

    /**
     * Connects and sets the search path.
     *
     * Establishes a connection to be used for this session, assigning it to
     * the instance variable 'connection'. In addition, sets the search
     * path to parlgov.
     *
     * @param  url       the url for the database
     * @param  username  the username to connect to the database
     * @param  password  the password to connect to the database
     * @return           true if connecting is successful, false otherwise
     */
    public abstract boolean connectDB(String url, String username, String password);

    /**
     * Closes the database connection.
     *
     * @return true if the closing was successful, false otherwise
     */
    public abstract boolean disconnectDB();

    /**
     * Returns the list of elections over the years in descending order of years
     * and the cabinets that have formed after each election.
     *
     * @param  countryName   name of the country
     * @return               a 2-D array with the first row being the list of elections
     *                       over the election dates in descending order of years and the second row
     *                       being the cabinets that have formed after each election.
     */
    public abstract ElectionCabinetResult electionSequence(String countryName);

    /**
     * Returns the list of politicians similar to a given politician.
     *
     * Does so by finding politicians that have similar comments and descriptions (combined)
     * using Jaccard similarity and a threshold.
     *
     * @param  politicianId   id of the politician
     * @param  threshold        Jaccard similarity threshold
     * @return                  a list of politicians with Jaccard similarity of
     *                          comments and descriptions above the given threshold
     */
    public abstract List<String> findSimilarPoliticians(int politicianId, float threshold);

}
