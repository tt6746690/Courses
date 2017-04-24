vlib work
vlog -timescale 1ns/1ns part3.v
vsim control
log {/*}
add wave {/*}

#/*
#  An FSM that takes (x, y) and returns
#  x_dir and y_dir, which is inverted if (x, y) is at edge of screen
#  */
#module control(
#  input go,
#  input resetn,
#  input clk,
#  input [7:0] pos_limit,  // the right / bottom limit to x / y respectively
#  input [7:0] coord,      // Current x or y coordinate
#  output reg dir          // 1 => positive direction  | 0 => Negative direction
#  );


# clock
force {clk} 0 0, 1 10 -repeat 20

# active low reset
force {resetn} 0
run 20ns
force {resetn} 1

# go
force {go} 1

# set pos_limit and coord
force {pos_limit} 00001111
force {coord} 00000000
run 30ns
force {coord} 00000010
run 30ns
force {coord} 00001110
run 30ns
force {coord} 00001111
run 30ns
force {coord} 00001100
run 30ns
force {coord} 00000001
run 30ns
force {coord} 00000000
run 30ns
force {coord} 00000011
run 30ns
