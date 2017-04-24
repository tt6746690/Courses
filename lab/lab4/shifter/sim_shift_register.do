vlib work
vlog -timescale 1ns/1ns shiftregister.v
vsim ShiftRegister
log {/*}
add wave {/*}

# Alternating clock
force {KEY[0]} 0 0, 1 20 -repeat 40

# default LoadVal 
force {SW[7:0]} 01101110

# resetting 
force {SW[9]} 0
run 40ns

force {SW[9]} 1

# default values for ASR, shiftright, Load_n 
 now defaults to parallel loading 
force {KEY[3:1]} 000
run 35ns

# Now begins Logical Right Shfit 
force {KEY[3:1]} 011
run 200ns 
