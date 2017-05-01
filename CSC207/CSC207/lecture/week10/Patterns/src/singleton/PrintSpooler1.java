package singleton;

/*
 * Approach: 
 * - private static final instance variable
 * - private constructor
 * - public static getInstance method
 */
public class PrintSpooler1 {
    
    private static final PrintSpooler1 instance = new PrintSpooler1();
    
    private PrintSpooler1() {}
    
    public static PrintSpooler1 getInstance() {
        return instance;
    }
    
}

/* Advantages: 
 *  - static instance : created only once
 *  - final: cannot be modified
 *  - public static getInstance() method provides a global point of access
 *
 * Disadvantages:
 *  - instance created when class PrintSpooler1 is loaded, that is
 *    before getInstance() is called for the first time
 */