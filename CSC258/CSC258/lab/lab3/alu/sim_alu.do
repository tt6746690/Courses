vlib work
vlog -timescale 1ns/1ns alu.v
vsim alu
log {/*}
add wave {/*}

# A = 1111  B = 0101
# Case1:
#                 --> HEX2 = F  HEX0 = 5
force {SW[7]} 1
force {SW[6]} 1
force {SW[5]} 1
force {SW[4]} 1
force {SW[3]} 0
force {SW[2]} 1
force {SW[1]} 0
force {SW[0]} 1

# Key = 0
# A + 1: 1111 + 1 --> ALUout[8:0] = 0001 0000;
#                 --> HEX5 = 1 HEX4 = 0
force {KEY[2]} 0
force {KEY[1]} 0
force {KEY[0]} 0
run 10ns

# Key = 1
# A + B: 1111 + 0101 --> ALUout[8:0] = 0001 0100;
#                    --> HEX5 = 1 HEX4 = 4
force {KEY[2]} 0
force {KEY[1]} 0
force {KEY[0]} 1
run 10ns


# Key = 2
# A + B: 1111 + 0101 --> ALUout[8:0] = 0001 0100;
#                    --> HEX5 = 1 HEX4 = 4
force {KEY[2]} 0
force {KEY[1]} 1
force {KEY[0]} 0
run 10ns


# Key = 3
# {A|B, A^B}         --> ALUout[8:0] = 1111 1010;
#                    --> HEX5 = F HEX4 = A
force {KEY[2]} 0
force {KEY[1]} 1
force {KEY[0]} 1
run 10ns


# Key = 4
# If any of 4 bits are high --> ALUout[8:0] = 0000 0001;
#                           --> HEX5 = 0 HEX4 = 1
force {KEY[2]} 1
force {KEY[1]} 0
force {KEY[0]} 0
run 10ns


# Key = 5
# Output the input   --> ALUout[8:0] = 1111 0101;
#                    --> HEX5 = F HEX4 = 5
force {KEY[2]} 1
force {KEY[1]} 0
force {KEY[0]} 1
run 10ns
