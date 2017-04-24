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
