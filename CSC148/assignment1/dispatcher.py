from driver import Driver
from rider import Rider
from location import Location

class Dispatcher:
    """A dispatcher fulfills requests from riders and drivers for a
    ride-sharing service.

    When a rider requests a driver, the dispatcher assigns a driver to the
    rider. If no driver is available, the rider is placed on a waiting
    list for the next available driver. A rider that has not yet been
    picked up by a driver may cancel their request.

    When a driver requests a rider, the dispatcher assigns a rider from
    the waiting list to the driver. If there is no rider on the waiting list
    the dispatcher does nothing. Once a driver requests a rider, the driver
    is registered with the dispatcher, and will be used to fulfill future
    rider requests.
    """

    def __init__(self):
        """Initialize a Dispatcher.

        @type self: Dispatcher
        @rtype: None
        """
        self.riderWaitlist = []
        self.driverRegistry = []

    def __str__(self):
        """Return a string representation.

        @type self: Dispatcher
        @rtype: str

        >>> d = Dispatcher()
        >>> d.riderWaitlist.append(Rider('Peter', Location(0,0), Location(1,1), 5))
        >>> d.driverRegistry.append(Driver('John', Location(5,5), 10))
        >>> print(d)
        rider Peter -> origin: (0, 0), destination: (1, 1), patience 5, status: waiting
        driver John -> location: (5, 5), is idle? True
        <BLANKLINE>
        """
        s = ''
        for i in self.riderWaitlist:
            s += i.__str__()
            s += '\n'
        for i in self.driverRegistry:
            s += i.__str__()
            s += '\n'
        return s

    def request_driver(self, rider):
        """Return a driver for the rider, or None if no driver is available.

        Add the rider to the waiting list if there is no available driver.

        @type self: Dispatcher
        @type rider: Rider
        @rtype: Driver | None

        >>> d = Dispatcher()
        >>> d.request_driver(Rider('Peter', Location(0,0), Location(1,1), 5))
        >>> d.riderWaitlist[0].id
        'Peter'
        >>> d.driverRegistry.append(Driver('John', Location(5,5), 10))
        >>> return_driver = d.request_driver(Rider('Peter', Location(0,0), Location(1,1), 5))
        >>> return_driver.id
        'John'
        """

        if len(self.driverRegistry) != 0:
            return min(self.driverRegistry, key = lambda driver: driver.get_travel_time(rider.origin))
        else:
            self.riderWaitlist.append(rider)
            return None


    def request_rider(self, driver):
        """Return a rider for the driver, or None if no rider is available.

        If this is a new driver, register the driver for future rider requests.

        @type self: Dispatcher
        @type driver: Driver
        @rtype: Rider | None

        >>> d = Dispatcher()
        >>> d.request_rider(Driver('John', Location(5,5), 10))

        >>> d.driverRegistry[0].id
        'John'
        >>> d.riderWaitlist.append(Rider('Peter', Location(0,0), Location(1,1), 5))
        >>> return_rider = d.request_rider(Driver('John', Location(5,5), 10))
        >>> return_rider.id
        'Peter'
        """

        self.driverRegistry.append(driver)
        if self.riderWaitlist:
            r = self.riderWaitlist[0]
            self.riderWaitlist.remove(r)
            return r
        else:
            return None

    def cancel_ride(self, rider):
        """Cancel the ride for rider.

        @type self: Dispatcher
        @type rider: Rider
        @rtype: None

        >>> d = Dispatcher()
        >>> d.riderWaitlist.append(Rider('Peter', Location(0,0), Location(1,1), 5))
        >>> d.riderWaitlist[0].id
        'Peter'
        >>> d.cancel_ride(Rider('Peter', Location(0,0), Location(1,1), 5))
        >>> len(d.riderWaitlist) == 0
        True
        """
        for e in self.riderWaitlist:
            if e == rider:
                self.riderWaitlist.remove(e)
