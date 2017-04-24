

module lab7_part3
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


  /*
    Connection to FPGA
    */
  wire resetn;
	assign resetn = KEY[0];

	wire [2:0] colour;
  assign colour = SW[9:7];    // added

  wire go;
  // assign go = ~KEY[3];        // load signal
  assign go = 1'b1;

  // x, y direction
  wire x_dir, y_dir;

  // current coordinate of the box, connects to FSM to enable edge detection
  wire [7:0] x;
  wire [7:0] y;

  // x and y coordinate of every cell in the box, connects to VGA
  wire [7:0] x_out;
  wire [7:0] y_out;

  /*
    Datapath stores and changes (x, y) with certain frame speed
    */
  datapath d0(
    .enable(go),
    .resetn(resetn),
    .clk(CLOCK_50),
    .colour(colour),
    .x_dir(x_dir),
    .y_dir(y_dir),
    .x(x_out),
    .y(y_out),
    .c_out(c_out)
    );


  /*
    Instansiate FSM for x & y coordinate direction
    */
  control x_c(
    .go(go),
    .resetn(resetn),
    .clk(CLOCK_50),
    .pos_limit(8'd155),
    .coord(x),
    .dir(x_dir)
    );
  control y_c(
    .go(go),
    .resetn(resetn),
    .clk(CLOCK_50),
    .pos_limit(8'd115),
    .coord(y),
    .dir(y_dir)
    );


  /*
    Use the vga adapter to draw appropriate (x_out, y_out) coordinate
    */
  vga_adapter VGA(
    .resetn(resetn),
    .clock(CLOCK_50),
    .colour(c_out),
    .x(x_out),
    .y(y_out),
    .plot(1'b1),
    /* Signals for the DAC to drive the monitor. */
    .VGA_R(VGA_R),
    .VGA_G(VGA_G),
    .VGA_B(VGA_B),
    .VGA_HS(VGA_HS),
    .VGA_VS(VGA_VS),
    .VGA_BLANK(VGA_BLANK_N),
    .VGA_SYNC(VGA_SYNC_N),
    .VGA_CLK(VGA_CLK));
    defparam VGA.RESOLUTION = "160x120";
    defparam VGA.MONOCHROME = "FALSE";
    defparam VGA.BITS_PER_COLOUR_CHANNEL = 1;
    defparam VGA.BACKGROUND_IMAGE = "black.mif";

endmodule



module datapath(
  input enable,
  input resetn,
  input clk,
  input [2:0] colour,
  input x_dir, y_dir,   // Direction of movement in x, y coordinates
  output [7:0] x_out,
  output [7:0] y_out,
  output [2:0] c_out
  );

  wire x, y;
  /*
    delays clk signal and sets proper enable for frame speed
    */
  wire d_enable;  // 60Hz enable = 1 frame
  wire f_enable;  // frame enable,

  delay_counter dc(
    .enable(enable),
    .clk(clk),
    .resetn(resetn),
    .d_enable(d_enable)
    );

  frame_counter fc(
    .enable(d_enable),
    .clk(clk),
    .resetn(resetn),
    .frame_per_pixel(6'b001111),  // (defaults to 15 frames) => 4 pixel / s
    .f_enable(f_enable)
    );

  /*
    Store and Increment/Decrement x & y coordinate with given
    x_dir and y_dir register, which stores direction to move
    */

  coord_counter x_cc(
    .enable(f_enable),
    .clk(clk),
    .resetn(resetn),
    .direction(x_dir),         // horizontal x - direction
    .init_pos(8'd0),           // (x, y) = (0, 60)
    .coord(x)                  // x coordinate
    );

  coord_counter y_cc(
    .enable(f_enable),
    .clk(clk),
    .resetn(resetn),
    .direction(y_dir),         // vertical y - direction
    .init_pos(8'd60),          // (x, y) = (0, 60)
    .coord(y)                  // y coordinate
    );

  /*
    Box counter generates (x_out, y_out) from (x, y) to plot the box
    */
  box_counter bc(
    .enable(f_enable),           // default enable signal
    .resetn(resetn),
    .clk(clk),                 // default clk signal
    .x(x),
    .y(y),
    .x_out(x_out),
    .y_out(y_out)
    );

endmodule

/*
  An FSM that takes (x, y) and returns
  x_dir and y_dir, which is inverted if (x, y) is at edge of screen
  */
module control(
  input go,
  input resetn,
  input clk,
  input [7:0] pos_limit,  // the right / bottom limit to x / y respectively
  input [7:0] coord,      // Current x or y coordinate
  output reg dir          // 1 => positive direction  | 0 => Negative direction
  );


  reg current_state, next_state;

  localparam POSITIVE_DIRECTION = 1'b0,
             NEGATIVE_DIRECTION = 1'b1;

  // state table for changing direction
  always @(*)
  begin: state_table
    case (current_state)
      POSITIVE_DIRECTION: next_state = (coord == pos_limit) ?           NEGATIVE_DIRECTION: POSITIVE_DIRECTION;
      NEGATIVE_DIRECTION: next_state = (coord == 0) ? POSITIVE_DIRECTION: NEGATIVE_DIRECTION;
    endcase
  end

  // output logic
  always @(*)
  begin: signals
    case (current_state)
      POSITIVE_DIRECTION: dir = 1'b1;
      NEGATIVE_DIRECTION: dir = 1'b0;
    endcase
  end

  // current_state registers
  always@(posedge clk)
  begin: state_FFs
      if(!resetn)
          current_state <= POSITIVE_DIRECTION;
      else
        if(go)
          current_state <= next_state;
  end
endmodule

/*
  draws a box with coordinate (x,y) at top left corner
  the size of the box is equivalent to size
  */
module box_counter(
  input enable,
  input resetn,
  input clk,
  input [7:0] x, y,
  output [7:0] x_out, y_out
  );

  // 4-bit counter that adds
  reg [3:0] q;
  always @(posedge clk)
  begin
    if (!resetn) begin
      q <= 0;
    end
    else if (enable) begin
      if (q == 4'b1111)
        q <= 0;
      else
        q <= q + 1;
    end
  end

  // creates 4x4 square sequentially as counter output iterates
  assign x_out = x + q[1:0];
  assign y_out = y + q[3:2];

endmodule



/*
  20-bit delay counter delays a 50MHz clk to a 60Hz enable
  d_enable is a delayed enable that is high every 1 / 60 s
  */
module delay_counter(
  input enable,
  input clk,
  input resetn,
  output d_enable
  );

  reg [19:0] q;

  always @(posedge clk)
  begin: delay_counter
    if(!resetn)
      q <= 0;
    else if (enable) begin
      if(q == 20'd833334)
        q <= 0;
      else
        q <= q + 1;
    end
  end

  // 1100_1011_0111_0011_0110=833334 -> cycles required for 60Hz given 50MHz clk
  assign d_enable = (q == 20'd833334) ? 1 : 0;
endmodule


/*
  4-bit frame counter counts number of 1 / 60s frame
  Assumes an input 60Hz enable signal
  */
module frame_counter(
  input enable,
  input clk,
  input resetn,
  input [5:0] frame_per_pixel,    // # of frame / pixel moved | range: [1 ~ 60]
  output f_enable
  );

  reg [3:0] q;

  always @(posedge clk)
  begin: frame_counter
    if(!resetn)
      q <= 0;
    else if(enable) begin
      if(q == frame_per_pixel)
        q <= 0;
      else
        q <= q + 1;
    end
  end

  assign f_enable = (q == frame_per_pixel) ? 1 : 0;
endmodule


/*
  coordiante counter stores the x and y coordinates,
  coord increments when direction is set to high, and decrements when direction is low. The 8-bit init_pos is loaded to the counter during reset.
  */
module coord_counter(
  input enable,
  input clk,
  input resetn,
  input direction,        // east (for x) and south (for y) is positive, i.e. 1
  input [7:0] init_pos,   // initial posotion for coord loaded to counter
  output reg [7:0] coord  // 8-bit x or y coordinate
  );

  always @(posedge clk)
  begin: coord_counter
    if(!resetn)
      coord <= init_pos;
    else if(enable) begin
      if(direction)
        coord <= coord + 1;
      else
        coord <= coord - 1;
    end
  end

endmodule
