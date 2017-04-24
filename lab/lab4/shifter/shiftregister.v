
module ShiftRegister(LEDR, SW, KEY);
    input [9:0] SW;
    input [3:0] KEY;    
    output [7:0] LEDR;
    
    Shifter s0(
        .LoadVal(SW[7:0]),
        .Load_n(KEY[1]),
        .ShiftRight(KEY[2]),
        .ASR(KEY[3]),
        .clk(KEY[0]),
        .reset_n(SW[9]),
        .Q(LEDR[7:0])
    );
endmodule 




module Shifter(LoadVal, Load_n, ShiftRight, ASR, clk, reset_n, Q);
    input [7:0] LoadVal;
    input Load_n, ShiftRight, ASR, clk, reset_n;
    output [7:0] Q;

    wire c0, c1, c2, c3, c4, c5, c6, c7, c8;    // connection between 1-bit shfiter
    
    assign c8 = ASR ? LoadVal[7] : 1'b0;          // If ASR=1: do arithmetic shift otherwise do logical shift

    ShifterBit sb7(             // instantiates 1-bit Shifter
        .in(c8),                // input to Shifter 
        .load_val(LoadVal[7]),  // the parallel load value 
        .load_n(Load_n),        // determines parallel loading (0 -> load)
        .shift(ShiftRight),     // determines if shift right   (1 -> shift right)
        .clock(clk),            // the clock signal 
        .reset_n(reset_n),      // the reset signal clears shifter data 
        .out(c7)
    );

    ShifterBit sb6(             // instantiates 1-bit Shifter
        .in(c7),                // input to Shifter 
        .load_val(LoadVal[6]),  // the parallel load value 
        .load_n(Load_n),        // determines parallel loading (0 -> load)
        .shift(ShiftRight),     // determines if shift right   (1 -> shift right)
        .clock(clk),            // the clock signal 
        .reset_n(reset_n),      // the reset signal clears shifter data 
        .out(c6)
    );

    ShifterBit sb5(             // instantiates 1-bit Shifter
        .in(c6),                // input to Shifter 
        .load_val(LoadVal[5]),  // the parallel load value 
        .load_n(Load_n),        // determines parallel loading (0 -> load)
        .shift(ShiftRight),     // determines if shift right   (1 -> shift right)
        .clock(clk),            // the clock signal 
        .reset_n(reset_n),      // the reset signal clears shifter data 
        .out(c5)
    );

    ShifterBit sb4(             // instantiates 1-bit Shifter
        .in(c5),                // input to Shifter 
        .load_val(LoadVal[4]),  // the parallel load value 
        .load_n(Load_n),        // determines parallel loading (0 -> load)
        .shift(ShiftRight),     // determines if shift right   (1 -> shift right)
        .clock(clk),            // the clock signal 
        .reset_n(reset_n),      // the reset signal clears shifter data 
        .out(c4)
    );

    ShifterBit sb3(             // instantiates 1-bit Shifter
        .in(c4),                // input to Shifter 
        .load_val(LoadVal[3]),  // the parallel load value 
        .load_n(Load_n),        // determines parallel loading (0 -> load)
        .shift(ShiftRight),     // determines if shift right   (1 -> shift right)
        .clock(clk),            // the clock signal 
        .reset_n(reset_n),      // the reset signal clears shifter data 
        .out(c3)
    );

    ShifterBit sb2(             // instantiates 1-bit Shifter
        .in(c3),                // input to Shifter 
        .load_val(LoadVal[2]),  // the parallel load value 
        .load_n(Load_n),        // determines parallel loading (0 -> load)
        .shift(ShiftRight),     // determines if shift right   (1 -> shift right)
        .clock(clk),            // the clock signal 
        .reset_n(reset_n),      // the reset signal clears shifter data 
        .out(c2)
    );

    ShifterBit sb1(             // instantiates 1-bit Shifter
        .in(c2),                // input to Shifter 
        .load_val(LoadVal[1]),  // the parallel load value 
        .load_n(Load_n),        // determines parallel loading (0 -> load)
        .shift(ShiftRight),     // determines if shift right   (1 -> shift right)
        .clock(clk),            // the clock signal 
        .reset_n(reset_n),      // the reset signal clears shifter data 
        .out(c1)
    );

    ShifterBit sb0(             // instantiates 1-bit Shifter
        .in(c1),                // input to Shifter 
        .load_val(LoadVal[0]),  // the parallel load value 
        .load_n(Load_n),        // determines parallel loading (0 -> load)
        .shift(ShiftRight),     // determines if shift right   (1 -> shift right)
        .clock(clk),            // the clock signal 
        .reset_n(reset_n),      // the reset signal clears shifter data 
        .out(c0)
    );

    assign Q = {c7, c6, c5, c4, c3, c2, c1, c0};
   
endmodule



module ShifterBit(in, load_val, load_n, shift, clock, reset_n, out);
    input in, load_val, load_n, shift, clock, reset_n;
    output out;

    wire data_m0_to_m1, data_m1_to_dff;

    mux2to1 m0(
        .x(out),
        .y(in),
        .s(shift),
        .m(data_m0_to_m1)
    );

    mux2to1 m1(                 // initiates 2nd multiplexer
        .x(load_val),           // the parallel load value, returned if load_n=0
        .y(data_m0_to_m1),
        .s(load_n),
        .m(data_m1_to_dff)      // output to flip-flop 
    );

    flipflop ff0(               // instantiates flip-flop 
        .d(data_m1_to_dff),     // input to flip-flop
        .q(out),                // output from flip-flop 
        .clock(clock),          // clock signal 
        .reset_n(reset_n)       // synchronous active low reset 
    );
endmodule 

module flipflop(d, q, clock, reset_n);
    input d, clock, reset_n;
    output q;

    reg q;
    
    always @(posedge clock)
    begin 
        if (reset_n == 1'b0)
            q <= 1'b0;
        else 
            q <= d;
    end 
endmodule 


module mux2to1(x, y, s, m);
    input x, y, s;      // select x when s=0; select y otherwise
    output m;           // output m 

    assign m = s ? y: x;
endmodule 
