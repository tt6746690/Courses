/*
  The game_clk initially makes sprite move DELAY_LOWER_LIM pixel / s
  then the game clock goes faster until sprite
  reached a top speed of DELAY_UPPER_LIM pixel / s
  */
module delay(
  input enable,
  input clk,        // CLOCK_50
  input resetn,
  output game_clk
  );


  localparam DELAY_LOWER_LIM = 30'd555555,
             DELAY_UPPER_LIM = 30'd100000,
             DECREMENT = 30'd1,
             DELAY_SLOW_CLK_LIM = 30'd5000;

  reg [29:0] delay_lim;


  delay_counter dc0(
    .enable(enable),
    .clk(clk),
    .resetn(resetn),
    .delay(delay_lim),
    .d_enable(game_clk)
    );

  wire delay_slow_clk;

  delay_counter dc1(
    .enable(enable),
    .clk(clk),
    .resetn(resetn),
    .delay(DELAY_SLOW_CLK_LIM),
    .d_enable(delay_slow_clk)
    );


  always @(posedge delay_slow_clk, resetn) begin
    if(!resetn) begin
      delay_lim <= DELAY_LOWER_LIM;
	 end
    else if(enable) begin
      if(delay_lim >= DELAY_UPPER_LIM)
        delay_lim <= delay_lim - DECREMENT;
    end
  end

endmodule



/*
  30-bit delay counter delays a 50MHz clk
  */
module delay_counter(
  input enable,
  input clk,
  input resetn,
  input [29:0] delay,
  output d_enable
  );

  reg [29:0] q;

  always @(posedge clk)
  begin: delay_counter
    if(!resetn)
      q <= 0;
    else if (enable) begin
      if(q >= delay)
        q <= 0;
      else
        q <= q + 1;
    end else
        q <= 0;

  end

  assign d_enable = (q == delay) ? 1 : 0;
endmodule
