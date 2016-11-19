class Registry:
    '''
    the class Registry contains information of runners, specifically their email and speed estimate.
    the class Registry is also able to add new runner to existing list of runners, retrieve runners in a given speed category
    and look up on a runner by its email to find which speed category he/she is in.

    >>> registry = Registry()
    >>> registry.addRunner({'email': 'cheese@gmail.com', 'speed': '<40'})
    >>> registry.addRunner({'email': 'mark@gmail.com', 'speed': '<30'})
    >>> registry.addRunner({'email': 'danny@gmail.com', 'speed': '<30'})
    >>> registry.getCategory()
    ['<20', '<30', '<40', '>40']
    >>> registry.getRunnersInCategory('<30')
    [{'email': 'mark@gmail.com', 'speed': '<30'}, {'email': 'danny@gmail.com', 'speed': '<30'}]
    >>> registry.getSpeedByEmail('danny@gmail.com')
    <30

    Attributes:
    ===========
    @type runners: list
        a list of runner info
    @type categories: dict
        a dictionary of runner info separated by categories

    '''
    def __init__(self):
        '''
        create a new registry containing empty list of runners
        and empty dict of categories

        @type self: Registry
        @rtype: None

        '''
        self.runners = []
        self.categories = {'<20': [], '<30': [], '<40': [], '>40': []}

    def __eq__(self, other):
        '''
        returns whether this Registry is equivalent to other

        @type self: Registry
        @type other: Registry | Any
        @rtype: bool

        >>> registry1 = Registry()
        >>> registry1.addRunner({'email': 'cheese@gmail.com', 'speed': '<40'})
        >>> registry2 = Registry()
        >>> registry2.addRunner({'email': 'mark@gmail.com', 'speed': '<30'})
        >>> registry3 = Registry()
        >>> registry3.addRunner({'email': 'cheese@gmail.com', 'speed': '<40'})
        >>> registry1 == registry2
        False
        >>> registry1 == registry3
        True
        '''
        return (type(self) == type(other) and self.runners == other.runners)

    def __str__(self):
        '''
        Returns a user-friendly representation of Registry itself

        @type self: Registry
        @rtype: str

        >>> registry = Registry()
        >>> registry.addRunner({'email': 'cheese@gmail.com', 'speed': '<40'})
        >>> registry.addRunner({'email': 'mark@gmail.com', 'speed': '<30'})
        >>> print(registry)
        Runner 1: (cheese@gmail.com, <40)
        Runner 2: (mark@gmail.com, <30)
        '''
        s = ""
        for index, value in enumerate(self.runners):
            s += "Runner {}: ({}, {}, {}) \n".format(index + 1, value['name'], value['email'], value['speed'])
        return s



    def addRunner(self, runner):
        '''
        add new runners to the self.runners list
        populate categories dictionary

        @type self: Registry
        @type runner: dict
        @rtype: None

        >>> registry = Registry()
        >>> registry.addRuner({'email': 'cheese@gmail.com', 'speed': '<40'})

        '''
        if('email' not in runner or 'speed' not in runner):
            raise Exception('missing required info')
        else:
            self.categorize(runner)
            self.runners.append(runner)

    def getCategory(self):
        '''
        get available speed category of the race in the registry

        @return: a list of category names in string
        @rtype: list

        >>> registry = Registry()
        >>> print(registry.getCategory())
        ['<20', '<30', '<40', '>40']

        '''
        return list(self.categories.keys())

    def getRunnersInCategory(self, category):
        '''
        get all runner dictionaries in a list given a provided speed category

        @param category: speed category
        @type category: string
        @return: a list of runners
        @rtype: list

        >>> registry = Registry()
        >>> registry.addRunner({'email': 'cheese@gmail.com', 'speed': '<40'})
        >>> registry.addRunner({'email': 'mark@gmail.com', 'speed': '<30'})
        >>> registry.addRunner({'email': 'danny@gmail.com', 'speed': '<30'})
        >>> registry.getRunnersInCategory('<30')
        [{'email': 'mark@gmail.com', 'speed': '<30'}, {'email': 'danny@gmail.com', 'speed': '<30'}]

        '''

        if (category not in self.categories):
            raise Exception('category not found')
        elif (not self.categories[category]):
            raise Exception('category empty')
        else:
            return self.categories[category]


    def getSpeedByEmail(self, email):
        '''
        get the speed category of the runner identified by a matching email

        @param email: email of runner to be looked up
        @type email: str
        @return: speed category of the runner
        @rtype: str

        >>> registry = Registry()
        >>> registry.addRunner({'email': 'cheese@gmail.com', 'speed': '<40'})
        >>> registry.getSpeedByEmail('cheese@gmail.com')
        <40

        '''

        speed = [d['speed'] for d in self.runners if d['email'] == email]  # list compression
        return speed[0] if speed else 'not found'




    def categorize(self, runner):
        '''
        sort and append runner into corresponding speed category under self.categories'

        @param runner: runner info
        @type runner: dict
        @rtype: None

        '''
        if(runner['speed'] == '<20'):
            self.categories['<20'].append(runner)
        if(runner['speed'] == '<30'):
            self.categories['<30'].append(runner)
        if(runner['speed'] == '<40'):
            self.categories['<40'].append(runner)
        if(runner['speed'] == '>40' ):
            self.categories['>40'].append(runner)
