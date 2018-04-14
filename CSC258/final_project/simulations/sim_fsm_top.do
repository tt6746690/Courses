vlib work
vlog -timescale 1ps/1ps ../fpga_top.v
vsim -L altera_mf_ver rundonaldrun
log {/*}
add wave {/*}



#wire go;
#assign go = ~KEY[3];
#wire resetn;
#assign resetn = KEY[0];
#assign enterkey = KEY[1];


force {KEY[3]} 0
force {KEY[1]} 0
force {CLOCK_50} 0 0, 1 5 -repeat 10


force {KEY[0]} 1
run 30ps
force {KEY[0]} 0
run 30ps
force {KEY[0]} 1
run 100ps
