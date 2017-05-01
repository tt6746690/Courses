vlib work
vlog -timescale 1ns/1ns flash.v
vsim RateCounter
log {/*}
add wave {/*}


# Alternating clock
force {r_clock} 0 0, 1 20 -repeat 40

# do reset first to give default values to bit stored in register
force {r_clear} 0;
run 2ns
force {r_clear} 1;
run 2ns

# choose speed determines value to load
force {load_val[27:0]} 0010111110101111000010000000

# load values now
force  {par_load} 1
run 20ns
force {par_load} 0

# enable r now
force {r_enable} 1
run 1200ns
