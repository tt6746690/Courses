vlib work
vlog -timescale 1ns/1ns ram32x4.v
vsim -L altera_mf_ver ram32x4
log {/*}
add wave {/*}

# address: address for read and write
# clock: clock signal
# data: data input for writing
# wren: write enable. 1 -> write, 0 -> read
# q: data output

# assign d_in = SW[3:0];
# assign addr = SW[8:4];
# assign enable = SW[9];
# assign clk = KEY[0];


# Write
force {clock} 0 0, 1 20 -repeat 40

force {wren} 0
force {address} 00000
force {data} 0000
run 40
force {wren} 1
force {address} 00001
force {data} 0001
run 40
force {address} 00010
force {data} 0010
run 40
force {wren} 0
force {address} 00011
force {data} 0011
run 40

force {clock} 0
run 40

# Read
force {clock} 0 0, 1 20 -repeat 40
force {wren} 0

force {address} 00000
run 40
force {address} 00001
run 40
force {address} 00010
run 40
force {address} 00011
run 40
