vlib work
vlog -timescale 1ps/1ps ../draw.v
vsim -L altera_mf_ver draw_background
log {/*}
add wave {/*}


# module draw_background(
#   input enable,
#   input clk,
#   input resetn,
#   input [7:0] x_in,           // upper left coordinate (x_in, y_in)
#   input [7:0] y_in,
#   input [7:0] width,          // dimension of rectangle to be drawn (width, height)
#   input [7:0] height,
#   output reg [7:0] x,         // outputs (x, y, color) every CLOCK_50
#   output reg [7:0] y,
#   output color
#  );


force {clk} 0 0, 1 1 -repeat 2

# draw 120x160 sprite at (0, 0)
force {x_in} 00000000
force {y_in} 00000000
force {width} 10100000
force {height} 01111000

# reset
force {resetn} 0
run 1ps
force {resetn} 1
run 1ps

# run 5x5=25 clk cycles
force {enable} 1
run 1000ps
