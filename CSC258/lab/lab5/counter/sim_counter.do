vlib work
vlog -timescale 1ns/1ns counter.v
vsim counter
log {/*}
add wave {/*}

# Assignments to the 8 bit counter
#    input KEY;
#    input [1:0] SW;
#    output [6:0] HEX0, HEX1;
#
#    wire clk, enable, clear_b;
#    assign clk = KEY;
#    assign enable = SW[1];
#    assign clear_b = SW[0];
#
#    wire [7:0] counter_out;


# Alternating clock
force {KEY} 0 0, 1 20 -repeat 40

# set initial value of counter_out to 0 with clear_b then switch back
force {SW[0]} 0
run 10ns
force {SW[0]} 1
run 10ns

# Start simulation by setting Enable = 1
force {SW[1]} 1
run 300ns

# should notice that Q[7:0] where the least significant digits
#     changes the fastest, and any subsequent digits change half as fast.
