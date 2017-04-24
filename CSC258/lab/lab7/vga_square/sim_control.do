vlib work
vlog -timescale 1ns/1ns part2.v
vsim control
log {/*}
add wave {/*}

# module control(
#  input go,
#  input resetn,
#  input clk,
#  output reg ld_x, ld_y
#  );



# clock
force {clk} 0 0, 1 10 -repeat 20

# active low reset
force {resetn} 0
run 20ns
force {resetn} 1
run 20ns


# simulating loading signal. ld_x and ld_y should be active staggered
force {go} 0 0, 1 30 -repeat 60
run 300ns
