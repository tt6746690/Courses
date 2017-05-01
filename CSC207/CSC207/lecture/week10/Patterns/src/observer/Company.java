package observer;

import java.util.Observable;
import java.util.Observer;

/**
 * A company.
 * @author campbell
 *
 */
class Company implements Observer {
    
    /** This Company's name. */
    private String name;
    
    /**
     * Constructs a new Company named name.
     * @param name This Company's name.
     */
    public Company(String name) {
        this.name = name;
    }

    @Override
    public void update(Observable o, Object arg) {
        System.out.println("Company " + this.name + 
                " observed a change in " + o);
        System.out.println("   The notification said: " + arg);
    }
    
    @Override
    public String toString() {
        return name;
    }
}