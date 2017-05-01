vlib work
vlog -timescale 1ns/1ns poly_function.v
vsim part2
log {/*}
add wave {/*}

# Sw[7:0] data_in

# KEY[0] synchronous reset when pressed
# KEY[1] go signal

# LEDR displays result
# HEX0 & HEX1 also displays result

# clock
force {clk} 0 0, 1 5 -repeat 10

# active low reset
force {resetn} 0
run 20ns
force {resetn} 1


# set input values for
  # A = 4
  # B = 2
  # C = 7
  # x = 3
# expected result for Ax^2 + Bx + C = 36 + 6 + 7 = 49 = 10'b00001_10001

force {data_in} 00000100
force {go} 1
run 20ns
force {go} 0
run 20ns

force {data_in} 00000010
force {go} 1
run 20ns
force {go} 0
run 20ns

force {data_in} 00000111
force {go} 1
run 20ns
force {go} 0
run 20ns

force {data_in} 00000011
force {go} 1
run 20ns
force {go} 0
run 20ns


force {go} 1
run 200ns
