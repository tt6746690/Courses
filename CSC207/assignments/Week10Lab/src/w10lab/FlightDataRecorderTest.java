package w10lab;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

//import org.junit.After;
//import org.junit.Before;
import org.junit.Test;


public class FlightDataRecorderTest {
	
	
	@Test 
	public void testAverageEmptyRecord(){
		FlightDataRecorder f = new FlightDataRecorder();
		
		double expected = 0.0;
		double result = f.average();
		
		assertEquals(expected, result, 0);
	}
	
	@Test 
	public void testAverageOne(){
		FlightDataRecorder f = new FlightDataRecorder();
		
		
		double a = 1.0;
		double b = 2.0;
		f.record(a);
		
		double expected = 1.0;
		double result = f.average();
		assertEquals(expected, result,0);
		
		f.record(b);
		
		double expected2  = 1.5;
		double result2 = f.average();
		assertEquals(expected2, result2, 0);
	}
	
	
	@Test
	public void testGetLastDataPoint(){
		FlightDataRecorder f = new FlightDataRecorder();
		double a = 1.0;
		double b = 2.0;
		double c = 3.0;
		f.record(a);
		f.record(b);
		f.record(c);

	
		List<Double> expected = new ArrayList<Double>(Arrays.asList(2.0, 3.0));
		List<Double> result = f.getLastDataPoints(2);
		assertEquals(expected, result);
	}
	
	@Test 
	public void testRecord(){
		FlightDataRecorder f = new FlightDataRecorder();
		List<Double> dlist = new ArrayList<Double>(Arrays.asList(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 ,8.0 ,9.0, 10.0, 11.0));
		for(double d: dlist){
			f.record(d);
		}
		
		assertEquals(dlist.subList(1, 11), f.getRecordedData().subList(1, 11));
		assertEquals(dlist.get(10), f.getRecordedData().get(0));

		
	}
	
	
	
	@Test
	public void test() {
		fail("Not yet implemented");
	}

}
