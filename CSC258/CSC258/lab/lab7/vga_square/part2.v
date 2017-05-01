

module part2
	(
		CLOCK_50,						//	On Board 50 MHz
		// Your inputs and outputs here
        KEY,
        SW,
		// The ports below are for the VGA output.  Do not change.
		VGA_CLK,   						//	VGA Clock
		VGA_HS,							//	VGA H_SYNC
		VGA_VS,							//	VGA V_SYNC
		VGA_BLANK_N,						//	VGA BLANK
		VGA_SYNC_N,						//	VGA SYNC
		VGA_R,   						//	VGA Red[9:0]
		VGA_G,	 						//	VGA Green[9:0]
		VGA_B   						//	VGA Blue[9:0]
	);

	input			CLOCK_50;				//	50 MHz
	input   [9:0]   SW;
	input   [3:0]   KEY;

	// Declare your inputs and outputs here
	// Do not change the following outputs
	output			VGA_CLK;   				//	VGA Clock
	output			VGA_HS;					//	VGA H_SYNC
	output			VGA_VS;					//	VGA V_SYNC
	output			VGA_BLANK_N;				//	VGA BLANK
	output			VGA_SYNC_N;				//	VGA SYNC
	output	[9:0]	VGA_R;   				//	VGA Red[9:0]
	output	[9:0]	VGA_G;	 				//	VGA Green[9:0]
	output	[9:0]	VGA_B;   				//	VGA Blue[9:0]

	wire resetn;
	assign resetn = KEY[0];

	// Create the colour, x, y and writeEn wires that are inputs to the controller.
	wire [2:0] colour;
  assign colour = SW[9:7];    // added
	wire [7:0] x;    // 256; x = 0 ~ 159
	wire [6:0] y;    // 128; 0 ~ 119
	wire writeEn;
  assign writeEn = ~KEY[1];   // added

  //
	// // Create an Instance of a VGA controller - there can be only one!
	// // Define the number of colours as well as the initial background
	// // image file (.MIF) for the controller.
	// vga_adapter VGA(
	// 		.resetn(resetn),
	// 		.clock(CLOCK_50),
	// 		.colour(colour),
	// 		.x(x),
	// 		.y(y),
	// 		.plot(plot),
	// 		/* Signals for the DAC to drive the monitor. */
	// 		.VGA_R(VGA_R),
	// 		.VGA_G(VGA_G),
	// 		.VGA_B(VGA_B),
	// 		.VGA_HS(VGA_HS),
	// 		.VGA_VS(VGA_VS),
	// 		.VGA_BLANK(VGA_BLANK_N),
	// 		.VGA_SYNC(VGA_SYNC_N),
	// 		.VGA_CLK(VGA_CLK));
	// 	defparam VGA.RESOLUTION = "160x120";
	// 	defparam VGA.MONOCHROME = "FALSE";
	// 	defparam VGA.BITS_PER_COLOUR_CHANNEL = 1;
	// 	defparam VGA.BACKGROUND_IMAGE = "black.mif";

	// Put your code here. Your code should produce signals x,y,colour and writeEn/plot
  wire go;
  assign go = ~KEY[3];        // load signal

	// for the VGA controller, in addition to any other functionality your design may require.

  wire ld_x, ld_y, plot;
  // Instansiate datapath
  datapath d0(
    .dataIn(SW[6:0]),
    .resetn(resetn),
    .enable(writeEn),
    .ld_x(ld_x),
    .ld_y(ld_y),
    .clk(CLOCK_50),
    .x_out(x),
    .y_out(y)
    );

  // Instansiate FSM control
  control c0(
    .go(go),
    .resetn(resetn),
    .clk(CLOCK_50),
    .ld_x(ld_x),
    .ld_y(ld_y),
    .plot(plot)
    );

endmodule



module datapath(
  input [6:0] dataIn,
  input resetn,
  input enable,
  input ld_x, ld_y,
  input clk,
  output [7:0] x_out,
  output [6:0] y_out
  );

  reg [7:0] x;
  reg [6:0] y;
  // register x and y and input logic
  always @(posedge clk) begin
    if (!resetn) begin
      x <= 7'd0;
      y <= 7'd0;
    end
    else begin
      if (ld_x)
        x <= {1'b0, dataIn};
      if (ld_y)
        y <= dataIn;
    end
  end

  // 4-bit counter that loops over itself
  reg [3:0] q;
  always @(posedge clk)
  begin
    if (!resetn) begin
      q <= 4'b0000;
    end
    else if (enable) begin
      if (q == 4'b1111)
        q <= 4'b0000;
      else
        q <= q + 1;
    end
  end

  // creates 4x4 square sequentially as counter output iterates
  assign x_out = x + q[1:0];
  assign y_out = y + q[3:2];

endmodule

module control(
  input go,
  input resetn,
  input clk,
  output reg ld_x, ld_y, plot
  );


  reg [2:0] current_state, next_state;

  localparam LOAD_X = 3'd0,
             LOAD_X_WAIT = 3'd1,
             LOAD_Y = 3'd2,
             LOAD_Y_WAIT = 3'd3,
             PLOT = 3'd4;

  // state table for loading X and Y into register
  always @(*)
  begin: state_table
    case (current_state)
      LOAD_X: next_state = go ? LOAD_X_WAIT: LOAD_X;
      LOAD_X_WAIT: next_state = go ? LOAD_X_WAIT: LOAD_Y;
      LOAD_Y: next_state = go ? LOAD_Y_WAIT: LOAD_Y;
      LOAD_Y_WAIT: next_state = go ? PLOT: LOAD_Y_WAIT;
      PLOT: next_state = go ? LOAD_X: PLOT;
        // do not return to LOAD_X
        // otherwise ld_x enabled and will reload x register with y's value
      default: next_state = LOAD_X;
    endcase
  end

  // output logic
  always @(*)
  begin: signals
    ld_x = 1'b0;
    ld_y = 1'b0;
    plot = 1'b0;
    case (current_state)
      LOAD_X: ld_x = 1;
      LOAD_Y: ld_y = 1;
      PLOT: plot = 1;
    endcase
  end

  // current_state registers
  always@(posedge clk)
  begin: state_FFs
      if(!resetn)
          current_state <= LOAD_X;
      else
          current_state <= next_state;
  end
endmodule
