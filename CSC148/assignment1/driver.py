from location import Location, manhattan_distance
from rider import Rider


class Driver:
    """A driver for a ride-sharing service.

    === Attributes ===
    @type id: str
        A unique identifier for the driver.
    @type location: Location
        The current location of the driver.
    @type is_idle: bool
        A property that is True if the driver is idle and False otherwise.
    """

    def __init__(self, identifier, location, speed):
        """Initialize a Driver.

        @type self: Driver
        @type identifier: str
        @type location: Location
        @type speed: int
        @rtype: None
        """
        self.id = identifier
        self.location = location
        self.speed = speed
        self.is_idle = True
        self.destination = None


    def __str__(self):
        """Return a string representation.

        @type self: Driver
        @rtype: str

        >>> d = Driver('Mark', Location(0,0), 10)
        >>> print(d)
        driver Mark -> location: (0, 0), is idle? True
        """
        return 'driver {} -> location: {}, is idle? {}'.format(self.id, self.location, self.is_idle)

    def __eq__(self, other):
        """Return True if self equals other, and false otherwise.

        @type self: Driver
        @rtype: bool

        >>> d = Driver('drvier1', Location(0, 0), 10)
        >>> s =  Driver('drvier1', Location(0, 0), 10)
        >>> d == s
        True
        """
        return (isinstance(other, Driver) and
                self.id == other.id and
                self.speed == other.speed and
                self.location == other.location and
                self.is_idle == other.is_idle)

    def get_travel_time(self, destination):
        """Return the time it will take to arrive at the destination,
        rounded to the nearest integer.

        @type self: Driver
        @type destination: Location
        @rtype: int

        >>> d = Driver('drvier1', Location(0, 0), 10)
        >>> d.get_travel_time(Location(5, 5))
        1
        >>> d.get_travel_time(Location(5, 6))
        1
        """
        return round(manhattan_distance(self.location, destination)/self.speed)

    def start_drive(self, location):
        """Start driving to the location and return the time the drive will take.

        @type self: Driver
        @type location: Location
        @rtype: int

        >>> d = Driver('drvier1', Location(0, 0), 10)
        >>> d.start_drive(Location(5, 5))
        1
        >>> d2 = Driver('drvier2', Location(0, 0), 10)
        >>> d2.start_drive(Location(5, 6))
        1
        """
        self.destination = location
        self.is_idle = False
        r = round(manhattan_distance(self.location, location)/self.speed)
        self.location = location
        return r

    def end_drive(self):
        """End the drive and arrive at the destination.

        Precondition: self.destination is not None.

        @type self: Driver
        @rtype: None
        """
        self.destination = None
        self.is_idle = True

    def start_ride(self, rider):
        """Start a ride and return the time the ride will take.

        @type self: Driver
        @type rider: Rider
        @rtype: int

        >>> d = Driver('drvier1', Location(0, 0), 10)
        >>> d.start_ride(Rider('rider1', Location(0, 0), Location(10, 10), 100))
        2
        """
        self.destination = rider.destination
        self.is_idle = False
        return round(manhattan_distance(self.location, self.destination)/self.speed)

    def end_ride(self):
        """End the current ride, and arrive at the rider's destination.

        Precondition: The driver has a rider.
        Precondition: self.destination is not None.

        @type self: Driver
        @rtype: None
        """
        self.location = self.destination
        self.destination = None
        self.is_idle = True

if __name__ == '__main__':
    import doctest
    doctest.testmod()
