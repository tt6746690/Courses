vlib work
vlog -timescale 1ns/1ns part3.v
vsim coord_counter
log {/*}
add wave {/*}

#/*
#  coordiante counter stores the x and y coordinates,
#  coord increments when direction is set to high, and decrements when direction # is low. The 8-bit init_pos is loaded to the counter during reset.
#  */
#module coord_counter(
#  input enable,
#  input clk,
#  input resetn,
#  input direction,        // east (for x) and south (for y) is positive, i.e. 1
#  input [7:0] init_pos,   // initial position for coord loaded to counter
#  output reg [7:0] coord  // 8-bit x or y coordinate
#  );

# clock
force {clk} 0 0, 1 10 -repeat 20
force {init_pos} 00001000


# active low reset
force {resetn} 0
run 20ns
force {resetn} 1

# Now coord should increment by 1 once every 4 cycles
force {enable} 0 0, 1 30 -repeat 40
force {direction} 1
run 200ns

# Now reverse to negative direction, coord should decrement
force {direction} 0
run 200ns

# return at starting position 00001000
