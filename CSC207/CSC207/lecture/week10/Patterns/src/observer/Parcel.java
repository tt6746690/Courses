package observer;

import java.util.Observable;

/**
 * An observable parcel with a tracking number and location.
 * @author campbell
 *
 */
public class Parcel extends Observable {
    
    /** This Parcel's tracking number. */
    private String trackingNumber;
    
    /** This Parcel's location. */
    private String location;
    
    /**
     * Constructs a new Parcel with tracking number trackingNumber and 
     * location location.
     * @param trackingNumber This Parcel's tracking number.
     * @param location This Parcel's location.
     */
    public Parcel(String trackingNumber, String location) {
        this.trackingNumber = trackingNumber;
        this.location = location;

    }
    
    @Override
    public String toString() {
        return "Parcel " + trackingNumber + ".";
    }

    /**
     * Sets this Parcel's location to newLocation and notifies its Observers.
     * @param newLocation This Parcel's new location.
     */
    public void updateLocation(String newLocation) {
        location = newLocation;
        setChanged();
        notifyObservers("Updated location to " + location + ".");
    }
}
