package singleton;

/*
 * Approach 2: 
 * - private constructor
 * - a nested "holder" class contains the instance
 * - getInstance() returns the instance stored in the holder class
 */
public class PrintSpooler2 {

  private PrintSpooler2() {}

  public static PrintSpooler2 getInstance() {
    return PrintSpooler2Holder.INSTANCE;
  }

  private static class PrintSpooler2Holder {
    private static final PrintSpooler2 INSTANCE = new PrintSpooler2();
  }
}

/* Advantages:
 *  As with option 1: 
 *  - static instance : created only once
 *  - final: cannot be modified
 *  - public static getInstance() method provides a global point of access
 *
 *  Addresses a problem of option 1: 
 *  - instance created when class PrintSpooler2Holder is loaded, which 
 *    happens when getInstance() is called for the first time
 *  - "initialization on demand"
 */