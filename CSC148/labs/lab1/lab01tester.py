from lab01 import Registry

race = Registry()
race.addRunner({'name': 'Gerhard', 'email': 'gerhard@gmail.com', 'speed': '<40'})
race.addRunner({'name': 'Tom', 'email': 'tom@gmail.com', 'speed': '<30'})
race.addRunner({'name': 'Toni', 'email': 'danny@gmail.com', 'speed': '<20'})
race.addRunner({'name': 'Margot', 'email': 'margot@gmail.com', 'speed': '<30' })
race.addRunner({'name': 'Gerhard', 'email': 'gerhard@gmail.com', 'speed': '<30'})


runners = race.getRunnersInCategory('<30')

s = ''
for idx, val in enumerate(runners, start=1):
    s += "runner {}: {} \n".format(idx, val['name'])
print(s)
