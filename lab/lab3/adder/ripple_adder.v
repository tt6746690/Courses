// SW[7:4] and SW[3:0] connects to inputs A and B
// SW[8] connects to cin
// LEDR[4] connects to cout; LEDR[3:0] connects to S

// 4 bit ripple adder 
module ripple_adder(LEDR, SW);
  input [9:0] SW;
  output [9:0] LEDR;

  wire c3, c2, c1;

  full_adder fa0(
      .S(LEDR[0]),
      .cout(c1),
      .A(SW[4]),
      .B(SW[0]),
      .cin(SW[8])
    );

  full_adder fa1(
      .S(LEDR[1]),
      .cout(c2),
      .A(SW[5]),
      .B(SW[1]),
      .cin(c1)
    );

  full_adder fa2(
      .S(LEDR[2]),
      .cout(c3),
      .A(SW[6]),
      .B(SW[2]),
      .cin(c2)
    );

  full_adder fa3(
      .S(LEDR[3]),
      .cout(LEDR[4]),
      .A(SW[7]),
      .B(SW[3]),
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
