vlib work
vlog -timescale 1ps/1ps ../draw.v
vsim -L altera_mf_ver draw_sprite
log {/*}
add wave {/*}

# module draw_sprite(           // 16x16 dimension using 256x1 ram
#  input enable,
#  input clk,
#  input resetn,
#  input [7:0] x_in,           // upper left coordinate (x_in, y_in)
#  input [7:0] y_in,
#  input [7:0] width,          // dimension of rectangle to be drawn (width, height)
#  input [7:0] height,
#  output [7:0] x,             // outputs (x, y, color) every CLOCK_50
#  output [7:0] y,
#  output color
#  );

force {clk} 0 0, 1 1 -repeat 2

# draw 16x16 sprite at (32, 64)
force {x_in} 00100000
force {y_in} 01000000
force {width} 00010000
force {height} 00010000

# reset
force {resetn} 0
run 1ps
force {resetn} 1
run 1ps

# run 5x5=25 clk cycles
force {enable} 1
run 600ps
