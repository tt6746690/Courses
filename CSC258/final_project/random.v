/*
  Generates a 1 bit random number generator
  where period is the average amount of time
  rand_bit to be high

  rate_inverse
  1 -> always 1
  2 -> 1 half the time
  65536 -> 1 every 65536
  */
module random(
  input enable,
  input clk,
  input resetn,
  input [15:0] seed,
  input [15:0] period,
  output reg rand_bit
  );

  wire [15:0] rand_16bit;

  linear_feedback_shifter lfb(
    .enable(enable),
    .clk(clk),
    .resetn(resetn),
    .seed(seed),
    .q(rand_16bit)
    );

  wire [15:0] test = rand_16bit % period;

  always @(posedge clk) begin
    if(!resetn)
      rand_bit <= 0;
    else if(enable && rand_16bit % period == 0)
      rand_bit <= 1;
    else
      rand_bit <= 0;
  end


endmodule

/*
  LFBS generates a pseudorandom 16 bit output q
  given a seed
  */
module linear_feedback_shifter(
    input enable,
    input clk,
    input resetn,
    input [15:0] seed,
    output reg [15:0] q
  );

  always @(posedge clk) begin
    if(!resetn)
      q <= seed;
    else if(enable)
      q <= (q<<1) + (((q[15] ^ q[13]) ^ q[12]) ^ q[10]);
  end

endmodule
