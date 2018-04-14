vlib work
vlog -timescale 1ps/1ps ../delay.v
vsim delay_counter
log {/*}
add wave {/*}

#module delay_counter(
#  input enable,
#  input clk,
#  input resetn,
#  input [29:0] delay,
#  output d_enable
#  );

force {delay} 10
force {enable} 0
force {clk} 0 0, 1 5ps -repeat 10ps

force {resetn} 1
run 30ps
force {resetn} 0
run 30ps
force {resetn} 1
run 30ps

force {enable} 1
run 500ps
