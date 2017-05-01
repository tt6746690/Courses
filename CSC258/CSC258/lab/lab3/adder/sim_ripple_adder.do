vlib work
vlog -timescale 1ns/1ns ripple_adder.v
vsim ripple_adder
log {/*}
add wave {/*}


# Case1: 1111 + 0001 --> LEDR[4] = 1 & LED[3:0] = 0000
force {SW[8]} 0
force {SW[7]} 1
force {SW[6]} 1
force {SW[5]} 1
force {SW[4]} 1
force {SW[3]} 0
force {SW[2]} 0
force {SW[1]} 0
force {SW[0]} 1


# Case2: 0011 + 0101 --> LEDR[4] = 0 & LED[3:0] = 1000
force {SW[8]} 0
force {SW[7]} 0
force {SW[6]} 0
force {SW[5]} 1
force {SW[4]} 1
force {SW[3]} 0
force {SW[2]} 1
force {SW[1]} 0
force {SW[0]} 1



# Case3: 1001 + 1101 --> LEDR[4] = 1 & LED[3:0] = 0110
force {SW[8]} 0
force {SW[7]} 1
force {SW[6]} 0
force {SW[5]} 0
force {SW[4]} 1
force {SW[3]} 1
force {SW[2]} 1
force {SW[1]} 0
force {SW[0]} 1
