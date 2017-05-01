vlib work
vlog -timescale 1ns/1ns part2.v
vsim datapath
log {/*}
add wave {/*}

# module datapath(
#   input [6:0] dataIn,
#   input resetn,
#   input enable,
#   input ld_x, ld_y,
#   input clk,
#   output [7:0] x_out,
#   output [6:0] y_out
#  );

# clock
force {clk} 0 0, 1 10 -repeat 20

# active low reset
force {resetn} 0
run 20ns
force {resetn} 1
run 20ns

# setting coordinate to (0, 0)
force {dataIn} 0000000
force {ld_x} 1
run 20ns
force {ld_x} 0

force {ld_y} 1
run 20ns
force {ld_y} 0

# starts the counter and observe x_out, y_out loops over 4x4 block
force {enable} 1
run 320ns
