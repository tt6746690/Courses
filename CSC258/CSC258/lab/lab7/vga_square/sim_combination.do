vlib work
vlog -timescale 1ns/1ns part2.v
vsim part2
log {/*}
add wave {/*}

#  assign colour = SW[9:7];    // added
#  assign resetn = KEY[0];
#  assign writeEn = ~KEY[1];   // added
#  assign go = ~KEY[3];
#  dataIn = SW[6:0]


# clock & enable = 0
force {CLOCK_50} 0 0, 1 10 -repeat 20
force {KEY[1]} 1

# active low reset
force {KEY[0]} 0
run 20ns
force {KEY[0]} 1
run 15ns

# load x and y to register
# active low load

force {SW[6:0]} 0100000
force {KEY[3]} 1
run 30ns
force {KEY[3]} 0
run 30ns
force {KEY[3]} 1


force {SW[6:0]} 0001000
run 30ns
force {KEY[3]} 0
run 30ns
force {KEY[3]} 1

# writeEnable
force {KEY[1]} 0
run 300ns
