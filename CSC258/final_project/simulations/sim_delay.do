vlib work
vlog -timescale 1ps/1ps ../delay.v
vsim delay
log {/*}
add wave {/*}

# hertz60 high every 3 clk cycle; period20 every 10; speed decrement every period20 high from 30
# game_clk high every speed * 60Hz delay so 2 * 3 = 6 base clk


#module delay(
#  input enable,
#  input clk,        // CLOCK_50
#  input resetn,
#  output game_clk
#  );


force {enable} 0
force {clk} 0 0, 1 5ps -repeat 10ps

force {resetn} 1
run 30ps
force {resetn} 0
run 30ps
force {resetn} 1
run 30ps
force {enable} 1
run 10000ps
