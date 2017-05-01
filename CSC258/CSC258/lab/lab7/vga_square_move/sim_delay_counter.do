vlib work
vlog -timescale 1ns/1ns part3.v
vsim delay_counter
log {/*}
add wave {/*}

#/*
#  20-bit delay counter delays a 50MHz clk to a 60Hz enable
#  d_enable is a delayed enable that is high every 1 / 60 s
#  */
#module delay_counter(
#  input enable,
#  input clk,
#  input resetn,
#  output d_enable
#  );

# clock
force {clk} 0 0, 1 10 -repeat 20

# active low reset
force {resetn} 0
run 20ns
force {resetn} 1

# go
force {enable} 1
run 1000ns
