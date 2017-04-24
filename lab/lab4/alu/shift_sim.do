
vlib work
vlog -timescale 1ns/1ns alu.v
vsim alu
log {/*}
add wave {/*}

# Alternating clock
force {KEY[0]} 0 0, 1 20 -repeat 40

# A = 0010      B defualts to 0000 i think
force {SW[3:0]} 2#0010
# reset = 1     Q <= ALUout
force {SW[9]} 1
force {SW[7:5]} 000
run 40ns
# set value B=0011 with A + 1 = 0000_0011 

# CASE5 
# B << A = 0000_1100
force {SW[7:5]} 2#101
run 30ns

# CASE6 
# B >> A = 0000_0011
force {SW[7:5]} 2#110
run 30ns


# CASE7
# B * A = 0011 * 0010 
force {SW[7:5]} 2#111
run 30ns
