vlib work
vlog -timescale 1ns/1ns alu.v
vsim alu
log {/*}
add wave {/*}

# Alternating clock
force {KEY[0]} 0 0, 1 20 -repeat 40

# A = 1101      B defualts to 0000 i think
force {SW[3:0]} 2#1101
# reset = 1     Q <= ALUout
force {SW[9]} 1
                            # HEX0 --> D

# CASE0
# A + 1 = 00001110 = 0x0_E
# HEX5 & HEX4 --> 0E changes only after 20ns because of the clock
force {SW[7:5]} 2#000
run 35ns


# CASE1
# A + B = 1101 + 1110 = 0001_1011 = 1_b hex display
# HEX5 and HEX4 change only at the last 5 ns
force {SW[7:5]} 2#001
run 30ns


# CASE2
# A + B = 1101 + 1011 = 0000_1000 = 0_8 hex display overflows
# HEX5 and 4 --> 25
force {SW[7:5]} 2#010
run 20ns


# now try resetting the value right before posedge
# upo posedge Q=00000000
# So alu updates AluOut = A + B = 1101 + 0000 = 0000_1101
force {SW[9]} 0
run 20ns

# now disable reset
force {SW[9]} 1
run 5ns

#Case3
# {A|B, A^B} = 1101_1101 = ALUout
force {SW[7:5]} 2#011
run 30ns
