vlib work
vlog -timescale 1ps/1ps ../fpga_top.v
vsim control
log {/*}
add wave {/*}

# module control(
#   input enable,
#   input clock,
#   input resetn,#
#
#   // USER CONTROL
#   input spacebar,#
#   input enterkey,
#
#   // TO DATAPATH
#   output reg [7:0] x,
#   output reg [7:0] y,
#   output reg [7:0] width,
#   output reg [8:0] height,
#   output reg [1:0] image_t
#   );

force {enable} 0
force {enterkey} 0
force {spacebar} 0
force {clock} 0 0, 1 5ps -repeat 10ps


force {resetn} 1
run 30ps
force {resetn} 0
run 50ps
force {resetn} 1
run 30ps

# enable: observe initial states, should be going over every init states, waiting for spacebar
force {enable} 1
run 300ps

# start game
force {enterkey} 1
run 20ps
force {enterkey} 0
run 500ps

# now jump
force {spacebar} 1
run 20ps
force {spacebar} 0
run 10000ps

force {enterkey} 1
run 20ps
force {enterkey} 0
run 300ps

# now jump
force {spacebar} 1
run 20ps
force {spacebar} 0
run 10000ps

# now jump
force {spacebar} 1
run 20ps
force {spacebar} 0
run 10000ps

# now jump
force {spacebar} 1
run 20ps
force {spacebar} 0
run 10000ps

#/* # press space when jumping */
#/* force {spacebar} 1 */
#/* run 20ps */
#/* force {spacebar} 0 */
#/* run 1000ps */
