vlib work
vlog -timescale 1ns/1ns flash.v
vsim flash_control
log {/*}
add wave {/*}



# Alternating clock
force {clock} 0 0, 1 20 -repeat 40

# do reset first to give default values to bit stored in register
force {dClear} 0;
force {rClear} 0;
run 2ns
force {dClear} 1;
force {rClear} 1;
run 2ns

# choose speed determines value to load
force {choose_speed[1:0]} 2#00

# load values now
force  {parLoad} 1
run 20ns
force {parLoad} 0

# enable r now
force {rEnable} 1
run 1200ns
