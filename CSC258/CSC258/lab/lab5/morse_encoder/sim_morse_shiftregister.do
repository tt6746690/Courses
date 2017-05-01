vlib work
vlog -timescale 1ns/1ns morse_encoder.v
vsim Shifter
log {/*}
add wave {/*}

# module shifter(s_out, s_load_val, shift_left, s_clock, s_clear);

# Alternating clock
force {s_clock} 0 0, 1 20 -repeat 40

# do reset first to give default values to bit stored in register
force {s_clear} 0
run 2ns
force {s_clear} 1
run 2ns

# assigns initial load_val for letter S = . . .  (3 dots)
force {s_load_val} 10101000000000

# now shift right on posedge of s_clock
force {shift_left} 1
run 200ns
