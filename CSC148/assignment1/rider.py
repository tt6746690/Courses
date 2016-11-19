"""
The rider module contains the Rider class. It also contains
constants that represent the status of the rider.

=== Constants ===
@type WAITING: str
    A constant used for the waiting rider status.
@type CANCELLED: str
    A constant used for the cancelled rider status.
@type SATISFIED: str
    A constant used for the satisfied rider status
"""
from location import Location

WAITING = "waiting"
CANCELLED = "cancelled"
SATISFIED = "satisfied"


class Rider:
    """A rider for a ride-sharing service.

    === Attributes ===
    @type id: str
        A unique identifier for the rider.
    @type destination: Location
        The destination for the rider
    @type status: str
        Rider's status may be one of waiting, cancelled, or satistied
    @type patience: int
        The number of time units the rider will wait to be picked up before they cancel their ride
    """

    def __init__(self, identifier, origin, destination, patience):
        """
        Initialize a rider

        status defaults to waiting once initialized

        @param Rider self: this rider
        @param str identifier: unique identifier of this rider
        @param Location origin: this rider's origin
        @param Location destination: this rider's destination
        @param int patience: The number of time units the rider will wait to be picked up before he cancel the ride
        @rtype: None
        """
        self.id = identifier
        self.origin = origin
        self.destination = destination
        self.status = WAITING
        self.patience = patience


    def __str__(self):
        """Return a string representation.

        @type self: Rider
        @rtype: str

        >>> r = Rider('Peter', Location(0, 0), Location(1, 1), 10)
        >>> print(r)
        rider Peter -> origin: (0, 0), destination: (1, 1), patience 10, status: waiting
        """
        return 'rider {} -> origin: {}, destination: {}, patience {}, status: {}'.format(self.id, self.origin, self.destination, self.patience, self.status)

    def __eq__(self, other):
        '''evaluate equivalence of rider objects

        @param Rider self: this rider object
        @param Rider | Any other: other rider object

        >>> r1 = Rider('Peter', Location(0, 0), Location(1, 1), 10)
        >>> r2 = Rider('Peter', Location(0, 0), Location(1, 1), 10)
        >>> r3 = Rider('Peter', Location(0, 1), Location(1, 1), 10)
        >>> r1 == r2
        True
        >>> r1 == r3
        False
        '''
        return (isinstance(other, Rider) and
                self.id == other.id and
                self.origin == other.origin and
                self.destination == other.destination and
                self.patience == other.patience)
