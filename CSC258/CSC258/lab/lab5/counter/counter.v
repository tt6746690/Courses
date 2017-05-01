
module counter(HEX0, HEX1, KEY, SW);
    input [3:0] KEY;
    input [1:0] SW;
    output [6:0] HEX0, HEX1;

    wire clk, enable, clear_b;
    assign clk = KEY[0];
    assign enable = SW[1];
    assign clear_b = SW[0];

    wire [7:0] counter_out;


    counter8bit c(
        .Q(counter_out),
        .Enable(enable),
        .Clock(clk),
        .Clear_b(clear_b)
      );

    decoder d1(     // HEX1 displays most significant 4 bit of counter output
        .hex(HEX1),
        .n(counter_out[7:4])
      );

    decoder d0(     // HEX0 displays least significant 4 bit of counter output
        .hex(HEX0),
        .n(counter_out[3:0])
      );

endmodule


module counter8bit(Q, Enable, Clock, Clear_b);
    /* An 8 bit counter
        Q: 8-bit output of counter value
        Enable: counts if 1; holds if 0
        Clock: synchronous clock signal
        Clear_b: asynchronous active-low clear, sets Q to 0 instantly
    */
    input Enable, Clock, Clear_b;
    output [7:0] Q;

    // Note AND all previous input is necessary as there might be
    //  cases where the Q[0] = 1 for a bit and Q[1] just on posdge.
    //   quartus counts Q[1] & Enable as 1 and updates upon clock instantaneously
    //    this is not desired as we want all Q before
    //    the current flipflop to be 1 to overflow to the next Q
    wire [7:0] T_in;    // inputs to T_flipflops
    assign T_in[0] = Enable;
    assign T_in[1] = Enable & Q[0];
    assign T_in[2] = Enable & Q[1] & Q[0];
    assign T_in[3] = Enable & Q[2] & Q[1] & Q[0];
    assign T_in[4] = Enable & Q[3] & Q[2] & Q[1] & Q[0];
    assign T_in[5] = Enable & Q[4] & Q[3] & Q[2] & Q[1] & Q[0];
    assign T_in[6] = Enable & Q[5] & Q[4] & Q[3] & Q[2] & Q[1] & Q[0];
    assign T_in[7] = Enable & Q[6] & Q[5] & Q[4] & Q[3] & Q[2] & Q[1] & Q[0];

    T_flipflop t0(
        .q(Q[0]),
        .t(T_in[0]),
        .clock(Clock),
        .clear(Clear_b)
      );

    T_flipflop t1(
        .q(Q[1]),
        .t(T_in[1]),      // AND Enable and previous T flipflop
                                //  to make a counter
        .clock(Clock),
        .clear(Clear_b)
      );
    T_flipflop t2(
        .q(Q[2]),
        .t(T_in[2]),
        .clock(Clock),
        .clear(Clear_b)
      );
    T_flipflop t3(
        .q(Q[3]),
        .t(T_in[3]),
        .clock(Clock),
        .clear(Clear_b)
      );
    T_flipflop t4(
        .q(Q[4]),
        .t(T_in[4]),
        .clock(Clock),
        .clear(Clear_b)
      );
    T_flipflop t5(
        .q(Q[5]),
        .t(T_in[5]),
        .clock(Clock),
        .clear(Clear_b)
      );
    T_flipflop t6(
        .q(Q[6]),
        .t(T_in[6]),
        .clock(Clock),
        .clear(Clear_b)
      );

    T_flipflop t7(
        .q(Q[7]),
        .t(T_in[7]),
        .clock(Clock),
        .clear(Clear_b)
      );

endmodule




module T_flipflop(q, t, clock, clear);
    // A T-type flip-flop that
    //    1. toggles on positive edge
    //    2. with an active-low asynchronous clear
    //      meaning that as soon as clear change from 1 to 0 (negedge)
    //      the flip-flop resets immediately
    input t, clock, clear;
    output q;

    reg q;
    wire d;

    assign d = q ^ t;   // XOR gate for enable toggling
                        //  If T = 0 --> D = Q_previous
                        //  If T = 1 --> D = ~Q_previous
    always @(posedge clock, negedge clear)
    begin
        if (clear == 1'b0)
            q <= 1'b0;
        else
            q <= d;
    end
endmodule



module decoder(hex, n);
input [3:0] n;
output [6:0] hex;

  assign hex[0] = ~n[3] & ~n[2] & ~n[1] & n[0] |
                       ~n[3] & n[2] & ~n[1] & ~n[0] |
                       n[3] & n[2] & ~n[1] & n[0] |
                       n[3] & ~n[2] & n[1] & n[0];

  assign hex[1] = n[3] & n[2] & ~n[0] |
                        ~n[3] & n[2] & ~n[1] & n[0] |
                        n[3] & n[1] & n[0] |
                        n[2] & n[1] & ~n[0];

  assign hex[2] = n[3] & n[2] & n[1] |
                        ~n[3] & ~n[2] & n[1] & ~n[0] |
                       n[3] & n[2] & ~n[0];

  assign hex[3] = ~n[3] & n[2] & ~n[1] & ~n[0] |
                       ~n[3] & ~n[2] & ~n[1] & n[0] |
                       n[2] & n[1] & n[0] |
                       n[3] & ~n[2] & n[1] & ~n[0];

  assign hex[4] = ~n[3] & n[0] |
                       ~n[2] & ~n[1] & n[0] |
                       ~n[3] & n[2] & ~n[1];

  assign hex[5] = ~n[3] & ~n[2] & ~n[1] & n[0] |
                       ~n[3] & n[1] & n[0] |
                       n[3] & n[2] & ~n[1] & n[0] |
                       ~n[3] & ~n[2] & n[1];

  assign hex[6] = ~n[3] & ~n[2] & ~n[1] |
                       ~n[3] & n[2] & n[1] & n[0] |
                       n[3] & n[2] & ~n[1] & ~n[0];

endmodule
