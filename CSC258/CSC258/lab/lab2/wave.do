# Set the working dir, where all compiled Verilog goes.
vlib work

# Compile all Verilog modules in mux.v to working dir;
# could also have multiple Verilog files.
# The timescale argument defines default time unit
# (used when no unit is specified), while the second number
# defines precision (all times are rounded to this value)
vlog -timescale 1ns/1ns mux.v

# Load simulation using mux as the top level simulation module.
vsim mux

# Log all signals and add some signals to waveform window.
log {/*}
# add wave {/*} would add all items in top level simulation module.
add wave {/*}

# First test case
# Set input values using the force command, signal names need to be in {} brackets.
force {SW[0]} 0
force {SW[1]} 0
force {SW[9]} 0
# Run simulation for a few ns.
run 10ns

# Second test case: change input values and run for another 10ns.
# SW[0] should control LED[0]
force {SW[0]} 0
force {SW[1]} 1
force {SW[9]} 0
run 10ns

# ...
# SW[0] should control LED[0]
force {SW[0]} 1
force {SW[1]} 0
force {SW[9]} 0
run 10ns

# SW[0] should control LED[0]
force {SW[0]} 1
force {SW[1]} 1
force {SW[9]} 0
run 10ns

# SW[1] should control LED[0]
force {SW[0]} 0
force {SW[1]} 0
force {SW[9]} 1
run 10ns

# SW[1] should control LED[0]
force {SW[0]} 0
force {SW[1]} 1
force {SW[9]} 1
run 10ns

# SW[1] should control LED[0]
force {SW[0]} 1
force {SW[1]} 0
force {SW[9]} 1
run 10ns

# SW[1] should control LED[0]
force {SW[0]} 1
force {SW[1]} 1
force {SW[9]} 1
run 10ns

