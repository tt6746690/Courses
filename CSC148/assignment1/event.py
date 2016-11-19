"""Simulation Events

This file should contain all of the classes necessary to model the different
kinds of events in the simulation.
"""
from rider import Rider, WAITING, CANCELLED, SATISFIED
from dispatcher import Dispatcher
from driver import Driver
from location import deserialize_location, Location
from monitor import Monitor, RIDER, DRIVER, REQUEST, CANCEL, PICKUP, DROPOFF


class Event:
    """An event.

    Events have an ordering that is based on the event timestamp: Events with
    older timestamps are less than those with newer timestamps.

    This class is abstract; subclasses must implement do().

    You may, if you wish, change the API of this class to add
    extra public methods or attributes. Make sure that anything
    you add makes sense for ALL events, and not just a particular
    event type.

    Document any such changes carefully!

    === Attributes ===
    @type timestamp: int
        A timestamp for this event.
    """

    def __init__(self, timestamp):
        """Initialize an Event with a given timestamp.

        @type self: Event
        @type timestamp: int
            A timestamp for this event.
            Precondition: must be a non-negative integer.
        @rtype: None

        >>> Event(7).timestamp
        7
        """
        self.timestamp = timestamp

    # The following six 'magic methods' are overridden to allow for easy
    # comparison of Event instances. All comparisons simply perform the
    # same comparison on the 'timestamp' attribute of the two events.
    def __eq__(self, other):
        """Return True iff this Event is equal to <other>.

        Two events are equal iff they have the same timestamp.

        @type self: Event
        @type other: Event
        @rtype: bool

        >>> first = Event(1)
        >>> second = Event(2)
        >>> first == second
        False
        >>> second.timestamp = first.timestamp
        >>> first == second
        True
        """
        return self.timestamp == other.timestamp

    def __ne__(self, other):
        """Return True iff this Event is not equal to <other>.

        @type self: Event
        @type other: Event
        @rtype: bool

        >>> first = Event(1)
        >>> second = Event(2)
        >>> first != second
        True
        >>> second.timestamp = first.timestamp
        >>> first != second
        False
        """
        return not self == other

    def __lt__(self, other):
        """Return True iff this Event is less than <other>.

        @type self: Event
        @type other: Event
        @rtype: bool

        >>> first = Event(1)
        >>> second = Event(2)
        >>> first < second
        True
        >>> second < first
        False
        """
        return self.timestamp < other.timestamp

    def __le__(self, other):
        """Return True iff this Event is less than or equal to <other>.

        @type self: Event
        @type other: Event
        @rtype: bool

        >>> first = Event(1)
        >>> second = Event(2)
        >>> first <= first
        True
        >>> first <= second
        True
        >>> second <= first
        False
        """
        return self.timestamp <= other.timestamp

    def __gt__(self, other):
        """Return True iff this Event is greater than <other>.

        @type self: Event
        @type other: Event
        @rtype: bool

        >>> first = Event(1)
        >>> second = Event(2)
        >>> first > second
        False
        >>> second > first
        True
        """
        return not self <= other

    def __ge__(self, other):
        """Return True iff this Event is greater than or equal to <other>.

        @type self: Event
        @type other: Event
        @rtype: bool

        >>> first = Event(1)
        >>> second = Event(2)
        >>> first >= first
        True
        >>> first >= second
        False
        >>> second >= first
        True
        """
        return not self < other

    def __str__(self):
        """Return a string representation of this event.

        @type self: Event
        @rtype: str
        """
        raise NotImplementedError("Implemented in a subclass")

    def do(self, dispatcher, monitor):
        """Do this Event.

        Update the state of the simulation, using the dispatcher, and any
        attributes according to the meaning of the event.

        Notify the monitor of any activities that have occurred during the
        event.

        Return a list of new events spawned by this event (making sure the
        timestamps are correct).

        Note: the "business logic" of what actually happens should not be
        handled in any Event classes.

        @type self: Event
        @type dispatcher: Dispatcher
        @type monitor: Monitor
        @rtype: list[Event]
        """
        raise NotImplementedError("Implemented in a subclass")


class RiderRequest(Event):
    """A rider requests a driver.

    === Attributes ===
    @type rider: Rider
        The rider.
    """

    def __init__(self, timestamp, rider):
        """Initialize a RiderRequest event.

        @type self: RiderRequest
        @type rider: Rider
        @rtype: None
        """
        super().__init__(timestamp)
        self.rider = rider

    def do(self, dispatcher, monitor):
        """Assign the rider to a driver or add the rider to a waiting list.
        If the rider is assigned to a driver, the driver starts driving to
        the rider.

        Return a Cancellation event. If the rider is assigned to a driver,
        also return a Pickup event.

        @type self: RiderRequest
        @type dispatcher: Dispatcher
        @type monitor: Monitor
        @rtype: list[Event]

        >>> d = Dispatcher()
        >>> m = Monitor()
        >>> events = RiderRequest(2, Rider('Peter', Location(0,0), Location(5,5), 20)).do(d,m)
        >>> type(events[0])
        <class 'event.Cancellation'>

        >>> d.driverRegistry.append(Driver('Mark', Location(5,5), 10))
        >>> events = RiderRequest(2, Rider('Peter', Location(0,0), Location(5,5), 20)).do(d,m)
        >>> type(events[0])
        <class 'event.Pickup'>
        >>> type(events[1])
        <class 'event.Cancellation'>
        """
        monitor.notify(self.timestamp, RIDER, REQUEST,
                       self.rider.id, self.rider.origin)

        events = []
        driver = dispatcher.request_driver(self.rider)
        if driver is not None:
            travel_time = driver.start_drive(self.rider.origin)
            events.append(Pickup(self.timestamp + travel_time, self.rider, driver))
        events.append(Cancellation(self.timestamp + self.rider.patience, self.rider))

        # print(self.__str__())
        return events


    def __str__(self):
        """Return a string representation of this event.

        @type self: RiderRequest
        @rtype: str

        >>> d = Dispatcher()
        >>> m = Monitor()
        >>> rq = RiderRequest(2, Rider('Peter', Location(0,0), Location(5,5), 20))
        >>> print(rq)
        2 -- Peter: Request a driver
        """
        return "{} -- {}: Request a driver".format(self.timestamp, self.rider.id)


class DriverRequest(Event):
    """A driver requests a rider.

    === Attributes ===
    @type driver: Driver
        The driver.
    """

    def __init__(self, timestamp, driver):
        """Initialize a DriverRequest event.

        @type self: DriverRequest
        @type driver: Driver
        @rtype: None
        """
        super().__init__(timestamp)
        self.driver = driver

    def do(self, dispatcher, monitor):
        """Register the driver, if this is the first request, and
        assign a rider to the driver, if one is available.

        If a rider is available, return a Pickup event.

        @type self: DriverRequest
        @type dispatcher: Dispatcher
        @type monitor: Monitor
        @rtype: list[Event]


        >>> d = Dispatcher()
        >>> m = Monitor()
        >>> events = DriverRequest(0, Driver('Peter', Location(0,0), 10)).do(d,m)
        >>> len(events) == 0
        True

        >>> d.riderWaitlist.append(Rider('Mark', Location(5,5), Location(10,10), 10))
        >>> events = DriverRequest(0, Driver('Peter', Location(0,0), 10)).do(d,m)
        >>> type(events[0])
        <class 'event.Pickup'>
        """
        # Notify the monitor about the request.

        # Request a rider from the dispatcher.
        # If there is one available, the driver starts driving towards the
        # rider, and the method returns a Pickup event for when the driver
        # arrives at the riders location.

        monitor.notify(self.timestamp, DRIVER, REQUEST,
                       self.driver.id, self.driver.location)

        events = []
        rider = dispatcher.request_rider(self.driver)
        if rider is not None:
            travel_time = self.driver.start_drive(rider.origin)
            events.append(Pickup(self.timestamp + travel_time, rider, self.driver))

        # print(self.__str__())
        return events

    def __str__(self):
        """Return a string representation of this event.

        @type self: DriverRequest
        @rtype: str

        >>> d = Dispatcher()
        >>> m = Monitor()
        >>> dq = DriverRequest(0, Driver('Peter', Location(0,0), 10))
        >>> print(dq)
        0 -- Peter: Request a rider
        """
        return "{} -- {}: Request a rider".format(self.timestamp, self.driver.id)


class Cancellation(Event):
    """
    changes the waiting rider to a canceled rider.
    No future event is scheduled

    === Attributes ===
    @type rider: Rider
        The rider.
    @type driver: Driver
        The driver.
    """
    def __init__(self, timestamp, rider):
        """
        Initialize a Cancellation event.

        @type self: Pickup
        @type rider: Rider
        @rtype: None
        """
        super().__init__(timestamp)
        self.rider = rider

    def do(self, dispatcher, monitor):
        """changes the waiting rider to a canceled rider.
        No future event is scheduled

        @type self: Cancellation
        @type dispatcher: Dispatcher
        @type monitor: Monitor
        @rtype: None
        """

        if self.rider.status == WAITING:
            monitor.notify(self.timestamp, RIDER, CANCEL,
                           self.rider.id, self.rider.origin)
            dispatcher.cancel_ride(self.rider)
            self.rider.status = CANCELLED
            # print(self.__str__())
        else:
            return None

    def __str__(self):
        """Return a string representation of this event.

        @type self: Cancellation
        @rtype: str

        >>> d = Dispatcher()
        >>> m = Monitor()
        >>> cancel = Cancellation(10, Rider('Peter', Location(0,0), Location(5,5), 10))
        >>> print(cancel)
        10 -- Peter: Cancel Request
        """
        return "{} -- {}: Cancel Request".format(self.timestamp, self.rider.id)


class Pickup(Event):
    """
    an event that sets the driver's location to the rider's location. If rider is still waiting,
    the driver starts the ride and the driver's destination becomes rider's destination, and also return a Dropoff event
    If rider cancelled, driver will request a new event immediately, and the driver has no destination

    === Attributes ===
    @type rider: Rider
        The rider.
    @type driver: Driver
        The driver.
    """
    def __init__(self, timestamp, rider, driver):
        """
        Initialize a Pickup event.

        @type self: Pickup
        @type rider: Rider
        @type driver: Driver
        @rtype: None
        """
        super().__init__(timestamp)
        self.rider = rider
        self.driver = driver

    def do(self, dispatcher, monitor):
        """sets driver's location to rider's location. if rider is waiting,
        the driver starts driving and driver's destintion becomes rider's destination
        append a Dropoff event is scheduled for time they arrive at rider's destination
        and rider becomes satisfied

        if rider cancelled, a new event for driver requesting a rider is scheduled immediately
        and the driver has no destination at the moment

        @type self: Pickup
        @type dispatcher: Dispatcher
        @type monitor: Monitor
        @rtype: list[Event]

        >>> d, m = Dispatcher(), Monitor()
        >>> rider = Rider('Mark', Location(5,5), Location(15,15), 10)
        >>> driver = Driver('Peter', Location(5,5), 10)
        >>> events = Pickup(0, rider, driver).do(d,m)
        >>> type(events[0])
        <class 'event.Dropoff'>

        >>> impatient_rider = Rider('Luke', Location(5,5), Location(10,10), 0)
        >>> impatient_rider.status = CANCELLED
        >>> events = Pickup(0, impatient_rider, driver).do(d,m)
        >>> type(events[0])
        <class 'event.DriverRequest'>
        """

        events = []
        self.driver.end_drive()

        if self.rider.status == CANCELLED:
            events.append(DriverRequest(self.timestamp, self.driver))
        elif self.rider.status == WAITING:
            monitor.notify(self.timestamp, RIDER, PICKUP,
                           self.rider.id, self.rider.origin)
            monitor.notify(self.timestamp, DRIVER, PICKUP,
                           self.driver.id, self.rider.origin)
            travel_time = self.driver.start_ride(self.rider)
            self.rider.status = SATISFIED
            events.append(Dropoff(self.timestamp + travel_time, self.rider, self.driver))

        # print(self.__str__())
        return events

    def __str__(self):
        """Return a string representation of this event.

        @type self: Pickup
        @rtype: str

        >>> d, m = Dispatcher(), Monitor()
        >>> rider = Rider('Mark', Location(5,5), Location(15,15), 10)
        >>> driver = Driver('Peter', Location(5,5), 10)
        >>> pu = Pickup(0, rider, driver)
        >>> print(pu)
        0 -- Peter: Pickup Mark
        """
        return "{} -- {}: Pickup {}".format(self.timestamp, self.driver.id, self.rider.id)

class Dropoff(Event):
    """
    Sets driver's location to rider's destination. leaves the rider satisfied.
    Schedule a DriverRequest immediately. The driver has no destination atm

    === Attributes ===
    @type rider: Rider
        The rider.
    @type driver: Driver
        The driver.
    """
    def __init__(self, timestamp, rider, driver):
        """
        Initialize a Dropoff event.

        @type self: Dropoff
        @type rider: Rider
        @type driver: Driver
        @rtype: None
        """
        super().__init__(timestamp)
        self.rider = rider
        self.driver = driver

    def do(self, dispatcher, monitor):
        """
        Sets driver's location to rider's destination. leaves the rider satisfied.
        Schedule a DriverRequest immediately. The driver has no destination atm

        @type self: Dropoff
        @type dispatcher: Dispatcher
        @type monitor: Monitor
        @rtype: list[Event]

        >>> d, m = Dispatcher(), Monitor()
        >>> rider = Rider('Mark', Location(5,5), Location(15,15), 10)
        >>> driver = Driver('Peter', Location(5,5), 10)
        >>> events = Dropoff(0, rider, driver).do(d,m)
        >>> type(events[0])
        <class 'event.DriverRequest'>
        """

        events = []
        self.driver.end_ride()
        events.append(DriverRequest(self.timestamp, self.driver))

        monitor.notify(self.timestamp, RIDER, DROPOFF,
                       self.rider.id, self.rider.destination)
        monitor.notify(self.timestamp, DRIVER, DROPOFF,
                       self.driver.id, self.driver.location)

        # print(self.__str__())
        return events

    def __str__(self):
        """Return a string representation of this event.

        @type self: Dropoff
        @rtype: str

        >>> d, m = Dispatcher(), Monitor()
        >>> rider = Rider('Mark', Location(5,5), Location(15,15), 10)
        >>> driver = Driver('Peter', Location(5,5), 10)
        >>> do = Dropoff(0, rider, driver)
        >>> print(do)
        0 -- Peter: Dropoff Mark
        """
        return "{} -- {}: Dropoff {}".format(self.timestamp, self.driver.id, self.rider.id)

def create_event_list(filename):
    """Return a list of Events based on raw list of events in <filename>.

    Precondition: the file stored at <filename> is in the format specified
    by the assignment handout.

    @param filename: str
        The name of a file that contains the list of events.
    @rtype: list[Event]

    >>> l = create_event_list('events_small.txt')
    >>> len(l)
    2
    >>> [print(event) for event in l]
    1 -- Dan: Request a driver
    10 -- Arnold: Request a rider
    [None, None]
    """
    events = []
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                # Skip lines that are blank or start with #.
                continue

            # Create a list of words in the line, e.g.
            # ['10', 'RiderRequest', 'Cerise', '4,2', '1,5', '15'].
            # Note that these are strings, and you'll need to convert some
            # of them to a different type.
            tokens = line.split()
            timestamp = int(tokens[0])
            event_type = tokens[1]

            # HINT: Use Location.deserialize to convert the location string to
            # a location.

            if event_type == "DriverRequest":
                id = tokens[2]
                location = deserialize_location(tokens[3])
                speed = int(tokens[4])
                # Create a DriverRequest event.
                event = DriverRequest(timestamp, Driver(id, location, speed))
            elif event_type == "RiderRequest":
                id = tokens[2]
                origin = deserialize_location(tokens[3])
                destination = deserialize_location(tokens[4])
                patience = int(tokens[5])
                # Create a RiderRequest event.
                event = RiderRequest(timestamp, Rider(id, origin, destination, patience))

            events.append(event)
    return events
