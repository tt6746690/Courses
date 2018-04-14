package w10lab;

import java.util.ArrayList;
import java.util.List;

/**
 * A representation of a Flight Data Recorder (FDR) of an aircraft.
 * Stores CAPACITY data points.
 */
public class FlightDataRecorder {

    /**
     * The capacity of FlightDataRecorders.
     */
    public static int CAPACITY = 10;

    /**
     * The data recorded in this FlightDataRecorder.
     */
    private final List<Double> dataPoints;

    /**
     * A new FlightDataRecorder with no data.
     */
    public FlightDataRecorder() {
        dataPoints = new ArrayList<>(CAPACITY);
    }

    /**
     * Returns the average of the data points currently recorded in this
     * FlightDataRecorder (i.e., excludes data points already overwritten).
     * Return 0.0 if no values have been recorded.
     *
     * @return The average of the data points recorded in this
     * FlightDataRecorder or 0.0 if no data points have been recorded.
     */
    public double average() {
        double sum = 0.0; // was 1 changed to 0 
        for(double d : dataPoints) {
          sum += d;
        }
        if(dataPoints.size()==0){		// set up if statement to catch for when size() = 0 prevents returning NaN
        	return sum;					// have to think about when size() = 0
        }
        return sum / dataPoints.size();
    }

    /**
     * If this FlightDataRecorder contains at least {@code len} data points,
     * returns a List of the most recent {@code len} data points. For example,
     * if your recorder contains {0, 1, 2, 3} and you ask for {@code
     * getLastDataPoints(2)}, this method would return {2, 3}.  Otherwise,
     * returns a List of all the data points that this FlightDataRecorder
     * stores. The List contains the recorded data points in reverse order:
     * the most recent item will be at index 0.
     *
     * @param len the number of recent data points to be included.
     * @return the last {@code len} data points.
     */
    public List<Double> getLastDataPoints(int len) {
        return dataPoints.subList(dataPoints.size() - len, dataPoints.size());  //change from  subList(1, len);
    }

    /**
     * Returns a List of all recorded data points.
     *
     * @return a List of all recorded data points.
     */
    public List<Double> getRecordedData() {
        return dataPoints;
    }

    /**
     * Records a new data point in this FlightDataRecorder.
     *
     * @param dataPoint The new data point to be recorded.
     */
    public void record(Double dataPoint) {
    	int i = dataPoints.size() - CAPACITY - 1;
        if(dataPoints.size() > CAPACITY) {
          dataPoints.remove(i);
        }
        dataPoints.add(i, dataPoint);
    }
}
