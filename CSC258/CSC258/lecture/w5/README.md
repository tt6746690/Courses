

Note

+ XOR + D flipflop ---> T flipflop
+ serial T flipflop with AND gates in between them --> Counter
+ mux2to1 + D flipflop ---> load register 

![](assets/README-a6564.png)


Propagation delay
+  clock-to-output delay
+ the time a flip-flop takes to change its output after the clock edge.


__Shift Register__
+ In digital circuits, a shift register is a cascade of flip flops, sharing the same clock, in which the output of each flip-flop is connected to the 'data' input of the next flip-flop in the chain, resulting in a circuit that shifts by one position the 'bit array' stored in it, 'shifting in' the data present at its input and 'shifting out' the last bit in the array, at each transition of the clock input.


1. Serial-in parallel-out (SIPO)
+ data is input serially, once data has been locked in, it may either read off at each output simultaneously or it can be shifted out.


2. Parallel-in Serial-out (PISO)
+ data input loads in one clock cycle


__Counter__

__Ripple (Async) Counter__
+  changing state bits are used as clocks to subsequent state flip-flops,
+ For any T flipflop added. one will get an additional 1 bit counter that counts half as fast.
+ counts to 2^n - 1 where n is the number of bits
+ suffer from unstable output as the overflows ripple from state to state.



__Synchronous Counter__
+ clock inputs of all flip-flops are connected together and are triggered by the input pulses. Hence all flip-flops change state simultaneously. (in parallel)


__Finite state machine__
