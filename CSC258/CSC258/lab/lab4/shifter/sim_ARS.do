vlib work
vlog -timescale 1ns/1ns shiftregister.v
vsim ShiftRegister
log {/*}
add wave {/*}

# Alternating clock
force {KEY[0]} 0 0, 1 20 -repeat 40

# default LoadVal 
force {SW[7:0]} 11101101

# resetting 
force {SW[9]} 0
run 40ns

force {SW[9]} 1

# default values for ASR, shiftright, Load_n 
# now defaults to parallel loading 
force {KEY[3:1]} 000
run 35ns

# now test what happens if ShiftRight=0
#       turns out that the data stay at the register; no change observed
force {KEY[3:1]} 101
run 60ns 

# now begins arithmetic right shift 
force {KEY[3:1]} 111
run 150ns
