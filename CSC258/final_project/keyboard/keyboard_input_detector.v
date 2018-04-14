/*
  Outputs 1 when the spacebar is pressed, 0 when the spacebar is not.
*/

module space_input_detector(
  input  CLOCK_50,
  input  PS2_DAT,
  input  PS2_CLK,
  input [7:0] ON_SIGNAL,
  input [7:0] OFF_SIGNAL,
  inout  [35:0]  GPIO_0, GPIO_1,
  output KEY_ON
  );

	//  set all inout ports to tri-state
	assign  GPIO_0    =  36'hzzzzzzzzz;
	assign  GPIO_1    =  36'hzzzzzzzzz;

	wire reset = 1'b0;
	wire [7:0] scan_code;

	reg [7:0] history[1:3];
	wire read, scan_ready;

	reg KEY_ON_REG;

	oneshot pulser(
		.pulse_out(read),
		.trigger_in(scan_ready),
		.clk(CLOCK_50)
	);

	keyboard kbd(
	  .keyboard_clk(PS2_CLK),
	  .keyboard_data(PS2_DAT),
	  .clock50(CLOCK_50),
	  .reset(reset),
	  .read(read),
	  .scan_ready(scan_ready),
	  .scan_code(scan_code)
	);

  always @(posedge scan_ready)
  begin
    // let go
    if (history[2] == OFF_SIGNAL) begin
      KEY_ON_REG <= 1'b1;
    end
    else if (history[2] == ON_SIGNAL) begin
      KEY_ON_REG <= 1'b0;
    end
  end

	always @(posedge scan_ready)
	begin
		 history[3] <= history[2];
		 history[2] <= history[1];
		 history[1] <= scan_code;
	end

  assign KEY_ON = KEY_ON_REG;

endmodule

module enter_input_detector(
  input  CLOCK_50,
  input  PS2_DAT,
  input  PS2_CLK,
  input [7:0] ON_SIGNAL,
  input [7:0] OFF_SIGNAL,
  inout  [35:0]  GPIO_0, GPIO_1,
  output KEY_ON
  );

	//  set all inout ports to tri-state
	assign  GPIO_0    =  36'hzzzzzzzzz;
	assign  GPIO_1    =  36'hzzzzzzzzz;

	wire reset = 1'b0;
	wire [7:0] scan_code;

	reg [7:0] history[1:3];
	wire read, scan_ready;

	reg KEY_ON_REG;

	oneshot pulser(
		.pulse_out(read),
		.trigger_in(scan_ready),
		.clk(CLOCK_50)
	);

	keyboard kbd(
	  .keyboard_clk(PS2_CLK),
	  .keyboard_data(PS2_DAT),
	  .clock50(CLOCK_50),
	  .reset(reset),
	  .read(read),
	  .scan_ready(scan_ready),
	  .scan_code(scan_code)
	);

  always @(posedge scan_ready)
  begin
    // let go
    if (history[2] == OFF_SIGNAL) begin
      KEY_ON_REG <= 1'b1;
    end
    else if (history[2] == ON_SIGNAL) begin
      KEY_ON_REG <= 1'b0;
    end
  end

	always @(posedge scan_ready)
	begin
		 history[3] <= history[2];
		 history[2] <= history[1];
		 history[1] <= scan_code;
	end

  assign KEY_ON = KEY_ON_REG;

endmodule

