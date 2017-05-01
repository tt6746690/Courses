`include "decoder.v"
`include "ripple_adder.v"


// SW[7:4] and SW[3:0] connects to inputs A and B
// KEY[2:0] connects to function input
// LEDR[7:0] connects to ALUout[7:0]
// Display A and B with HEX2 and HEX0 with HEX1 and HEX3 set to 0
// HEX4 and HEX5 should display ALUout
module alu(LEDR, HEX0, HEX1, HEX2, HEX3, HEX4, HEX5, KEY, SW);

  input [2:0] KEY;
  input [9:0] SW;
  output [6:0] HEX0, HEX1, HEX2, HEX3, HEX4, HEX5;
  output [9:0] LEDR;

  wire [9:0] SW;
  wire [9:0] LEDR;
  reg [7:0] ALUout;

  wire [4:0] fn0_out;
  wire [4:0] fn1_out;
  wire [4:0] fn2_out;
  wire [7:0] fn3_out;
  wire [7:0] fn4_out;
  wire [7:0] fn5_out;

  ripple_adder ra0( // = A + 1
      .carry_sum(fn0_out),  // 5-bit
      .two_number({{SW[7:4]}, 4'b0001}) // 8-bit
    );

  ripple_adder ra1( // = A + B
      .carry_sum(fn1_out),
      .two_number(SW[7:0])
    );

  assign fn2_out = SW[7:4] + SW[3:0];  // built in addition
  assign fn3_out = {{SW[7:4] | SW[3:0]}, {SW[7:4] ^ SW[3:0]}};
  assign fn4_out = ( | SW[7:0] == 1'b0 ) ? 8'b00000000 : 8'b00000001;
  // {7}
  assign fn5_out = {{SW[7:4]}, {SW[3:0]}};

  always @(*)
  begin
    case (KEY)
      0: ALUout = {3'b000, fn0_out}; // sign extend 5-bit fn0_out
      1: ALUout = {3'b000, fn1_out};
      2: ALUout = {3'b000, fn2_out};
      3: ALUout = fn3_out;
      4: ALUout = fn4_out;
      5: ALUout = fn5_out;
      default: ALUout = 8'b00000000;
    endcase
  end

  assign LEDR = ALUout; // Conversionof 8 bit to 10 bit preserve value


    // set up 7-segment display
    decoder d0(
        .hex(HEX0),
        .switch(SW[3:0])
      );
    decoder d1(
        .hex(HEX1),
        .switch(4'b0000)
      );
    decoder d2(
        .hex(HEX2),
        .switch(SW[7:4])
      );
    decoder d3(
        .hex(HEX3),
        .switch(4'b0000)
      );

    decoder d4(
        .hex(HEX4),
        .switch(ALUout[3:0])
      );
    decoder d5(
        .hex(HEX5),
        .switch(ALUout[7:4])
      );


endmodule
