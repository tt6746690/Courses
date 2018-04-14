vlib work
vlog -timescale 1ps/1ps ../fpga_top.v
vsim -L altera_mf_ver datapath
log {/*}
add wave {/*}

#module datapath(
#  input clock,
#  input resetn,
#
#  // FROM CONTROL
#  input [7:0] x_from_control,
#  input [7:0] y_from_control,
#  input [7:0] width_from_control,
#  input [7:0] height_from_control,
#  input [1:0] object_choice,
#
#  // TO VGA
#  output reg [7:0] x_out,
#  output reg [7:0] y_out,
#  output reg [0:0] colour_out
#  );

force {enable} 0
force {clock} 0 0, 1 5ps -repeat 10ps

force {resetn} 1
run 30ps
force {resetn} 0
run 30ps
force {resetn} 1
run 100ps

# draw sprite at (0, 0)
force {x_from_control} 00010000
force {y_from_control} 00100000
force {width_from_control} 00010000
force {height_from_control} 00010000
force {object_choice} 10

force {enable} 1
run 2000ps
