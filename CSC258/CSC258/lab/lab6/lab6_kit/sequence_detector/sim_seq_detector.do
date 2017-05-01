vlib work
vlog -timescale 1ns/1ns sequence_detector.v
vsim sequence_detector
log {/*}
add wave {/*}

# SW[0] reset when 0
# SW[1] input signal (w)

# KEY[0] clock signal

# LEDR[2:0] displays current state
# LEDR[9] displays output


# Alternating clock    high low high low ...
force {KEY[0]} 0 0, 1 10 -repeat 20

# reset first
force {SW[0]} 0
run 30ns
# input signal w
force {SW[0]} 1
force {SW[1]} 0
run 45ns
force {SW[1]} 1
run 90ns
force {SW[1]} 0
run 20ns
force {SW[1]} 1
run 20ns
force {SW[1]} 0
run 60ns
