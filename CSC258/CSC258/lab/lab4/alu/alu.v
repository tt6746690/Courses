
// SW[7:4] and SW[3:0] connects to inputs A and B
// KEY[2:0] connects to function input
// LEDR[7:0] connects to ALUout[7:0]
// Display A and B with HEX2 and HEX0 with HEX1 and HEX3 set to 0
// HEX4 and HEX5 should display ALUout
module alu(LEDR, HEX0, HEX1, HEX2, HEX3, HEX4, HEX5, KEY, SW);
  input [2:0] KEY;
  input [9:0] SW;
  output [6:0] HEX0, HEX1, HEX2, HEX3, HEX4, HEX5;
  output [7:0] LEDR;

  // alu
  wire [3:0] A, B;             // Input A
  wire [2:0] ALU_function;  // Specifies which ALU function to process
  reg [7:0] ALUout;         // Output

  assign A = SW[3:0];
  assign ALU_function = SW[7:5];
  assign LEDR = ALUout;     // Display alu output on LEDR

  wire [4:0] fn0_out;
  wire [4:0] fn1_out;

  ripple_adder ra0( // = A + 1
      .carry_sum(fn0_out),  // 5-bit
      .two_number({A, 4'b0001}) // 8-bit
    );

  ripple_adder ra1( // = A + B
      .carry_sum(fn1_out),
      .two_number({A, B})
    );

  always @(*)
  begin
    case (ALU_function)
      3'b000: ALUout = {3'b000, fn0_out};          // sign extend 5-bit fn0_out
      3'b001: ALUout = {3'b000, fn1_out};
      3'b010: ALUout = {3'b000, {A + B}};
      3'b011: ALUout = {{A | B}, {A ^ B}};         //  A OR B in upper bit; A XOR B in lower bit
      3'b100: ALUout = {7'b0000000, | {A, B}};    // outputs 1 if  has 1 in SW[7:0]
      3'b101: ALUout = B << A;                     // left shift B by A bits
      3'b110: ALUout = B >> A;                     // right shift B by A bits
      3'b111: ALUout = A * B;                      // A X B
      default: ALUout = 8'b00000000;          // connects to ground
    endcase
  end

  // 8-bit register:
  //    + positive-edge triggered, active-low synchronous reset
  //    + stores value of ALUout
  reg [7:0] Q;              // register output
  wire clock;               // Control for D latch
  wire reset;               // Reset for D latch
  assign clock = KEY[0];
  assign reset = SW[9];

  always @(posedge clock)
  begin
    if (reset == 1'b0)
      Q <= 8'b00000000;
    else
      Q <= ALUout;
  end
  assign B = Q[3:0];        // least significant 4 bit of register output assigned to B
                            // If = ALUout[3:0] instead the signal from ALU will not stabilize
                            //  which will introduce maximum iteration reached error

  // 7-segment display
  decoder d0(   // HEX0 displays A which is SW[3:0]
      .hex(HEX0),
      .switch(A)
    );
  decoder d4(   // least significant 4-bit of register
      .hex(HEX4),
      .switch(Q[3:0])
    );
  decoder d5(   // most significant 4-bit of register
      .hex(HEX5),
      .switch(Q[7:4])
    );
endmodule



//switch[3:0] data inputs
//hex[6:0] output display
module decoder(hex, switch);
  input [3:0] switch;
  output [6:0] hex;

  assign hex[0] = ~switch[3] & ~switch[2] & ~switch[1] & switch[0]  |
                   ~switch[3] & switch[2]  & ~switch[1] & ~switch[0] |
                   switch[3]  & ~switch[2] & switch[1] & switch[0] |
                   switch[3]  & switch[2]  & ~switch[1] & switch[0];

  assign hex[1] = switch[3] & switch[1] & switch[0] |
                  switch[2] & switch[1] & ~switch[0] |
                  switch[3] & switch[2] & ~switch[0] |
                  ~switch[3] & switch[2] & ~switch[1] & switch[0];

  assign hex[2] = ~switch[3] & ~switch[2] & switch[1] & ~switch[0] |
                 switch[3]  & switch[2]  & ~switch[0] |
                 switch[3]  & switch[2]  & switch[1];

  assign hex[3] = ~switch[3] & switch[2] & ~switch[1] & ~switch[0] |
                 ~switch[2] & ~switch[1] & switch[0] |
                 switch[3] & switch[2] & switch[1] & switch[0] |
                 switch[3] & ~switch[2] & switch[1] & ~switch[0];

  assign hex[4] = ~switch[3] & switch[0] |
                ~switch[2] & ~switch[1] & switch[0] |
                ~switch[3] & switch[2] & ~switch[1] & ~switch[0];

  assign hex[5] =  switch[3] & switch[2] & ~switch[1] & switch[0] |
                    ~switch[3] & ~switch[2] & switch[0] |
                    ~switch[3] & ~switch[2] & switch[1] |
                    ~switch[3] & switch[1] & switch[0];

  assign hex[6] = ~switch[3] & ~switch[2] & ~switch[1] |
                  switch[3] & switch[2] & ~switch[1] & ~switch[0] |
                  ~switch[3] & switch[2] & switch[1] & switch[0];

endmodule


// SW[7:4] and SW[3:0] connects to inputs A and B
// SW[8] connects to cin
// LEDR[4] connects to cout; LEDR[3:0] connects to S

// 4 bit ripple adder
module ripple_adder(carry_sum, two_number);
  input [7:0] two_number;
  output [4:0] carry_sum;

  wire c3, c2, c1;

  full_adder fa0(
      .S(carry_sum[0]),
      .cout(c1),
      .A(two_number[4]),
      .B(two_number[0]),
      .cin(1'b0)
    );

  full_adder fa1(
      .S(carry_sum[1]),
      .cout(c2),
      .A(two_number[5]),
      .B(two_number[1]),
      .cin(c1)
    );

  full_adder fa2(
      .S(carry_sum[2]),
      .cout(c3),
      .A(two_number[6]),
      .B(two_number[2]),
      .cin(c2)
    );

  full_adder fa3(
      .S(carry_sum[3]),
      .cout(carry_sum[4]),
      .A(two_number[7]),
      .B(two_number[3]),
      .cin(c3)
    );
endmodule


// one bit full adder
// S = sum ; cout = carry out ; cin = carry in
module full_adder(S, cout, A, B, cin);
  input A, B, cin;
  output S, cout;

  wire A, B, cin, cout, S;

  assign S = A ^ B ^ cin;
  assign cout = (A & B) | (A & cin) | (B & cin);
endmodule
