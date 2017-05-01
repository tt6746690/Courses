package observer;

import java.util.Observable;
import java.util.Observer;

/**
 *     A customer.
 * @author campbell
 */
public class Customer implements Observer {
    
    /** This Customer's name. */
    private String name;
    
    /**
     * Constructs a new Customer named named.
     * @param name The new Customer's name.
     */
    public Customer(String name) {
        this.name = name;
    }
    
    @Override
    public void update(Observable o, Object arg) {
        System.out.println("Customer " + this.name 
                + " observed a change in " + o);
        System.out.println("   The notification said: " + arg);
    }
    
    @Override
    public String toString() {
        return name;
    }

}
