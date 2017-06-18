import sys

marker_fp = open("marker", "r")
line = marker_fp.readline()
data = line.split()
marker_start = data[0][2:]
marker_end = data[1].strip()[2:]

#print "MARKER START: " + marker_start
#print "MARKER END: " + marker_end

start = False

for line in sys.stdin:
    	# ignore all valgrind comment lines
	if line[0] == '=':
		continue
	
	data1 = line.split()
	data2 = data1[1].split(',')
	addr = int(data2[0], 16)
	instr = data1[0]

	# ignore all the memory accesses before the marker_start address
	if not start and addr != int(marker_start, 16):
		continue
	else:
		start = True

	# ignore everything after the marker_end address
	if addr == int(marker_end, 16):
		exit(0)

	print hex(addr) + "," + instr
