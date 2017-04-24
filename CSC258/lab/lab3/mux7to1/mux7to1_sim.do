vlib work
vlog -timescale 1ns/1ns mux7to1.v
vsim mux7to1
log {/*}
add wave {/*}

force {SW[9]} 0
force {SW[8]} 0 0, 1 100 -repeat 200
force {SW[7]} 0 0, 1 50 -repeat 100

force {SW[6]} 0 0, 1 50 -repeat 100
force {SW[5]} 0 0, 1 50 -repeat 100
force {SW[4]} 0 0, 1 50 -repeat 100
force {SW[3]} 0 0, 1 50 -repeat 100
force {SW[2]} 0 0, 1 50 -repeat 100
force {SW[1]} 0 0, 1 50 -repeat 100
force {SW[0]} 0 0, 1 50 -repeat 100
