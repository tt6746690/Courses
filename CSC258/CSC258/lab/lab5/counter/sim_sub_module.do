vlib work
vlog -timescale 1ns/1ns counter.v
vsim counter8bit
log {/*}
add wave {/*}



# Alternating clock
force {Clock} 0 0, 1 20 -repeat 40

# set initial value of counter_out to 0 with clear_b then switch back
force {Clear_b} 0
run 5ns
force {Clear_b} 1
run 5ns

# Start simulation by setting Enable = 1
force {Enable} 1
run 1500ns

# should notice that Q[7:0] where the least significant digits
#     changes the fastest, and any subsequent digits change half as fast.
