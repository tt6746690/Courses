vlib work
vlog -timescale 1ps/1ps ../random.v
vsim linear_feedback_shifter
log {/*}
add wave {/*}


#module linear_feedback_shifter(
#  input enable,
#  input clk,
#  input resetn,
#  input [15:0] seed,
#  output reg q
#  );

force {enable} 0
force {resetn} 1
force {seed} 1010101010101010
force {clk} 0 0, 1 10 -repeat 20

# reset
run 27ps
force {resetn} 0
run 24ps
force {resetn} 1
run 26ps

force {enable} 1
run 2000ps
