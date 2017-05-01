vlib work
vlog -timescale 1ns/1ns poly_function.v
vsim control
log {/*}
add wave {/*}

# Sw[7:0] data_in

# KEY[0] synchronous reset when pressed
# KEY[1] go signal

# LEDR displays result
# HEX0 & HEX1 also displays result

# stet up clock
force {clk} 0 0, 1 10 -repeat 20

# active low reset
force {resetn} 0
run 20ns
force {resetn} 1

run 20ns

force {go} 0 0, 1 20 -repeat 40
