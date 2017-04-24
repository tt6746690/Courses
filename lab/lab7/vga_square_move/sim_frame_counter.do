vlib work
vlog -timescale 1ns/1ns part3.v
vsim frame_counter
log {/*}
add wave {/*}


#/*
#  4-bit frame counter counts number of 1 / 60s frame
#  Assumes an input 60Hz enable signal
#  */
#module frame_counter(
#  input enable,
#  input clk,
#  input resetn,
#  input [5:0] frame_per_pixel,    // # of frame / pixel moved | range: [1 ~ 60]
#  output f_enable
#  );

# clock
force {clk} 0 0, 1 10 -repeat 20

# active low reset
force {resetn} 0
run 20ns
force {resetn} 1
# go
force {enable} 1


# test 4 frame per pixel, means high once every 4 clock cycle 
force {frame_per_pixel} 000100
run 300ns
