from location import manhattan_distance, Location

"""
The Monitor module contains the Monitor class, the Activity class,
and a collection of constants. Together the elements of the module
help keep a record of activities that have occurred.

Activities fall into two categories: Rider activities and Driver
activities. Each activity also has a description, which is one of
request, cancel, pickup, or dropoff.

=== Constants ===
@type RIDER: str
    A constant used for the Rider activity category.
@type DRIVER: str
    A constant used for the Driver activity category.
@type REQUEST: str
    A constant used for the request activity description.
@type CANCEL: str
    A constant used for the cancel activity description.
@type PICKUP: str
    A constant used for the pickup activity description.
@type DROPOFF: str
    A constant used for the dropoff activity description.
"""

RIDER = "rider"
DRIVER = "driver"

REQUEST = "request"
CANCEL = "cancel"
PICKUP = "pickup"
DROPOFF = "dropoff"


class Activity:
    """An activit`y that occurs in the simulation.

    === Attributes ===
    @type timesta`mp: int
        The time at which the activity occurred.
    @type description: str
        A description of the activity.
    @type identifier: str
        An identifier for the person doing the activity.
    @type location: Location
        The location at which the activity occurred.
    """

    def __init__(self, timestamp, description, identifier, location):
        """Initialize an Activity.

        @type self: Activity
        @type timestamp: int
        @type description: str
        @type identifier: str
        @type location: Location
        @rtype: None
        """
        self.description = description
        self.time = timestamp
        self.id = identifier
        self.location = location


class Monitor:
    """A monitor keeps a record of activities that it is notified about.
    When required, it generates a report of the activities it has recorded.
    """

    # === Private Attributes ===
    # @type _activities: dict[str, dict[str, list[Activity]]]
    #       A dictionary whose key is a category, and value is another
    #       dictionary. The key of the second dictionary is an identifier
    #       and its value is a list of Activities.

    def __init__(self):
        """Initialize a Monitor.

        @type self: Monitor
        """
        self._activities = {
            RIDER: {},
            DRIVER: {}
        }
        """@type _activities: dict[str, dict[str, list[Activity]]]"""

    def __str__(self):
        """Return a string representation.

        @type self: Monitor
        @rtype: str

        >>> m = Monitor()
        >>> m.notify(2, RIDER, CANCEL, 'mark', Location(0,4))
        >>> print(m)
        Monitor (0 drivers, 1 riders)
        """
        return "Monitor ({} drivers, {} riders)".format(
                len(self._activities[DRIVER]), len(self._activities[RIDER]))

    def notify(self, timestamp, category, description, identifier, location):
        """Notify the monitor of the activity.

        @type self: Monitor
        @type timestamp: int
            The time of the activity.
        @type category: DRIVER | RIDER
            The category for the activity.
        @type description: REQUEST | CANCEL | PICKUP | DROP_OFF
            A description of the activity.
        @type identifier: str
            The identifier for the actor.
        @type location: Location
            The location of the activity.
        @rtype: None
        """
        if identifier not in self._activities[category]:
            self._activities[category][identifier] = []

        activity = Activity(timestamp, description, identifier, location)
        self._activities[category][identifier].append(activity)


    def report(self):
        """Return a report of the activities that have occurred.

        @type self: Monitor
        @rtype: dict[str, object]

        >>> m = Monitor()
        >>> m.notify(1, RIDER, REQUEST, 'Dan', Location(1,1))
        >>> m.notify(10, DRIVER, REQUEST, 'Arnold', Location(3,3))
        >>> m.notify(12, DRIVER, PICKUP, 'Arnold', Location(1,1))
        >>> m.notify(12, RIDER, PICKUP, 'Dan', Location(1,1))
        >>> m.notify(17, DRIVER, DROPOFF, 'Arnold', Location(6,6))
        >>> m.notify(17, RIDER, DROPOFF, 'Dan', Location(6,6))
        >>> m.notify(17, DRIVER, REQUEST, 'Arnold', Location(6,6))
        >>> dict = m.report()
        >>> print(dict['driver_total_distance'])
        14.0
        >>> print(dict['rider_wait_time'])
        11.0
        >>> print(dict['driver_ride_distance'])
        10.0
        """

        return {"rider_wait_time": self._average_wait_time(),
                "driver_total_distance": self._average_total_distance(),
                "driver_ride_distance": self._average_ride_distance()}

    def _average_wait_time(self):
        """Return the average wait time of riders that have either been picked
        up or have cancelled their ride.

        @type self: Monitor
        @rtype: float

        >>> m = Monitor()
        >>> m.notify(1, RIDER, REQUEST, 'Dan', Location(1,1))
        >>> m.notify(10, DRIVER, REQUEST, 'Arnold', Location(3,3))
        >>> m.notify(12, DRIVER, PICKUP, 'Arnold', Location(1,1))
        >>> m.notify(12, RIDER, PICKUP, 'Dan', Location(1,1))
        >>> m.notify(17, DRIVER, DROPOFF, 'Arnold', Location(6,6))
        >>> m.notify(17, RIDER, DROPOFF, 'Dan', Location(6,6))
        >>> m.notify(17, DRIVER, REQUEST, 'Arnold', Location(6,6))
        >>> m._average_wait_time()
        11.0
        """
        wait_time = 0
        count = 0
        for activities in self._activities[RIDER].values():
            # A rider that has less than two activities hasn't finished
            # waiting (they haven't cancelled or been picked up).
            if len(activities) >= 2:
                # The first activity is REQUEST, and the second is PICKUP
                # or CANCEL. The wait time is the difference between the two.
                wait_time += activities[1].time - activities[0].time
                count += 1
        return wait_time / count

    def _average_total_distance(self):
        """Return the average distance drivers have driven.
        total distance by the number of drivers
        @type self: Monitor
        @rtype: float

        >>> m = Monitor()
        >>> m.notify(1, RIDER, REQUEST, 'Dan', Location(1,1))
        >>> m.notify(10, DRIVER, REQUEST, 'Arnold', Location(3,3))
        >>> m.notify(12, DRIVER, PICKUP, 'Arnold', Location(1,1))
        >>> m.notify(12, RIDER, PICKUP, 'Dan', Location(1,1))
        >>> m.notify(17, DRIVER, DROPOFF, 'Arnold', Location(6,6))
        >>> m.notify(17, RIDER, DROPOFF, 'Dan', Location(6,6))
        >>> m.notify(17, DRIVER, REQUEST, 'Arnold', Location(6,6))
        >>> m._average_total_distance()
        14.0
        """
        distance = 0
        for activities in self._activities[DRIVER].values():
            location = activities[0].location
            for e in activities:
                destination = e.location
                distance += manhattan_distance(location, destination)
                location = destination
        return distance / len(self._activities[DRIVER])

    def _average_ride_distance(self):
        """Return the average distance drivers have driven on rides.

        @type self: Monitor
        @rtype: float

        >>> m = Monitor()
        >>> m.notify(1, RIDER, REQUEST, 'Dan', Location(1,1))
        >>> m.notify(10, DRIVER, REQUEST, 'Arnold', Location(3,3))
        >>> m.notify(12, DRIVER, PICKUP, 'Arnold', Location(1,1))
        >>> m.notify(12, RIDER, PICKUP, 'Dan', Location(1,1))
        >>> m.notify(17, DRIVER, DROPOFF, 'Arnold', Location(6,6))
        >>> m.notify(17, RIDER, DROPOFF, 'Dan', Location(6,6))
        >>> m.notify(17, DRIVER, REQUEST, 'Arnold', Location(6,6))
        >>> m._average_ride_distance()
        10.0
        """
        distance = 0
        for activities in self._activities[RIDER].values():
            location = activities[0].location
            for e in activities:
                destination = e.location
                distance += manhattan_distance(location, destination)
                location = destination
        return distance / len(self._activities[DRIVER])
