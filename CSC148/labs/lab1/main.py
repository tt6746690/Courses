from lab01 import Registry

registry = Registry()
registry.addRunner({'email': 'cheese@gmail.com', 'speed': '<40'})
registry.addRunner({'email': 'mark@gmail.com', 'speed': '<30'})
registry.addRunner({'email': 'danny@gmail.com', 'speed': '<30'})
print(registry.getCategory())
# ['<20', '<30', '<40', '>40']
print(registry.getRunnersInCategory('<30'))
# [{'email': 'mark@gmail.com', 'speed': 23}, {'email': 'danny@gmail.com', 'speed': 27}]
print(registry.getSpeedByEmail('danny@gmail.com'))
# <30
print(registry)

registry1 = Registry()
registry1.addRunner({'email': 'cheese@gmail.com', 'speed': '<40'})
registry2 = Registry()
registry2.addRunner({'email': 'mark@gmail.com', 'speed': '<30'})
registry3 = Registry()
registry3.addRunner({'email': 'cheese@gmail.com', 'speed': '<40'})
print(registry1 == registry2)
# false
print(registry1 == registry3)
# true
