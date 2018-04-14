vlib work
vlog -timescale 1ps/1ps ../random.v
vsim random
log {/*}
add wave {/*}


#module random(
#  input enable,
#  input clk,
#  input resetn,
#  input [15:0] seed, period,
#  output reg rand_bit
#  );

force {enable} 0
force {clk} 0 0, 1 2 -repeat 4
force {resetn} 1
force {seed} 1010101010101010
force {period} 0000000000000001

# reset
force {resetn} 0
run 10ps
force {resetn} 1
run 10ps

force {enable} 1

# seed = 0 expect rand_bit = 1
run 200ps

# seed = 1 expect rand_bit =  1 about half of the time
force {period} 0000000000000011
run 2000ps


# seed = 65535 random_bit should be 1 about once every 2^16
force {period} 1111111111111111
run 2000ps
