//`include "delay.v";
//`include "rams/background_ram.v";
//`include "rams/sprite_ram.v";
//`include "rams/obstacle_ram.v";
//`include "rams/gameover_ram.v";

module rundonaldrun(
  input CLOCK_50,
  input [9:0] SW,
  input [3:0] KEY,
  input PS2_DAT,
  input PS2_CLK,

  inout [35:0] GPIO_0, GPIO_1,

  output VGA_CLK,
  output VGA_HS,
  output VGA_VS,
  output VGA_BLANK_N,
  output VGA_SYNC_N,
  output [9:0] VGA_R,
  output [9:0] VGA_G,
  output [9:0] VGA_B,
  output [6:0] LEDR,
  output [6:0] HEX0, HEX1, HEX2, HEX3, HEX4, HEX5
  );


  // Pressing down KEY[3] starts the entire game
  // KEY[0] resets the game
  wire go;
  assign go = SW[0];
  wire resetn;
  assign resetn = KEY[0];

  // main controller for game: spacebar
  wire spacebar;
  wire enterkey;
  assign enterkey = ~KEY[1];

  // interpreted for vga_adapter
  wire [7:0] x_into_datapath, y_into_datapath; // from FSM to datapath
  wire [7:0] width_into_datapath, height_into_datapath; // from FSM to datapath

  // going into vga_adapter
  wire [7:0] x, y; // the output x, y coordinate from draw

  // TODO: Need to change colour to allow for more than one colour
  wire [0:0] colour; // the colour to draw

  /* which object to load:
     00 - background_sprite   - draws the background where sprite at given x, y
     01 - sprite              - draws the sprite at a given x, y
     10 - background_obstacle - draws the background where obstacle at given x, y
     11 - obstacle            - draws the obstacle at a given x, y
  */
  wire [1:0] object_choice;

  localparam SPACEBAR_OFF = 8'b11110000,
				 SPACEBAR_ON = 8'b00101001,
				 ENTER_OFF = 8'b11110000,
				 ENTER_ON = 8'b1011010;

  // Detects whether the space was pressed
  space_input_detector space_detector(
    // inputs
    .CLOCK_50(CLOCK_50),
    .PS2_DAT(PS2_DAT),
    .PS2_CLK(PS2_CLK),
	 .ON_SIGNAL(SPACEBAR_ON),
	 .OFF_SIGNAL(SPACEBAR_OFF),
    .GPIO_0(GPIO_0),
    .GPIO_1(GPIO_1),
    .KEY_ON(spacebar)
  );

  // detects whether the enter key was pressed
//  enter_input_detector enter_detector(
//    .CLOCK_50(CLOCK_50),
//    .PS2_DAT(PS2_DAT),
//    .PS2_CLK(PS2_CLK),
//	 .ON_SIGNAL(ENTER_ON),
//	 .OFF_SIGNAL(ENTER_OFF),
//    .GPIO_0(GPIO_0),
//    .GPIO_1(GPIO_1),
//    .KEY_ON(LEDR[1])
//  );

  // VGA Module draws the background on init
  vga_adapter VGA(
  .resetn(resetn),
   .clock(CLOCK_50),
   .colour(colour),
   .x(x),
   .y(y),
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
    defparam VGA.BACKGROUND_IMAGE = "./mif/background.mono.mif";

  wire [7:0] hex_data;
  wire [7:0] sprite_data;
  wire [7:0] score;

  // the rundonaldrun's control module
  control c0(
    .enable(go),
    .clock(CLOCK_50),
    .resetn(resetn),
    .spacebar(spacebar),                      // controls the FSM
    .enterkey(enterkey),
    .x(x_into_datapath),          // x input into datapath (where to draw)
    .y(y_into_datapath),          // y input into datapath (where to draw)
    .width(width_into_datapath),
    .height(height_into_datapath),
    .image_t(object_choice),                     // what to draw for datapath (which mif to use)
	 .game_clock(LEDR[0]),
	 .ob_x(hex_data),
	 .sp_y(sprite_data),
	 .score(score),
	 .obs_out(LEDR[3]),
	 .generate_obs(LEDR[4])
    );

  hex_7seg(hex_data[3:0], HEX0);
  hex_7seg(hex_data[7:4], HEX1);

  hex_7seg(sprite_data[3:0], HEX2);
  hex_7seg(sprite_data[7:4], HEX3);

  hex_7seg(score[3:0], HEX4);
  hex_7seg(score[7:4], HEX5);

  // rundonaldrun's datapath
  datapath d0(
    .enable(go),
    .clock(CLOCK_50),
    .resetn(resetn),
    .x_from_control(x_into_datapath),    // tells where to draw
    .y_from_control(y_into_datapath),    // tells where to draw
    .width_from_control(width_into_datapath),
    .height_from_control(height_into_datapath),
    .object_choice(object_choice),               // tells what to draw (which mif to use)

    .x_out(x),          // direct x coord line into vga
    .y_out(y),          // direct y coord line into vga
    .colour_out(colour)                      // direct colour line into vga
    );

endmodule



/************/
/* CONTROL */
/***********/

/*
  The control module is responsible for handling the state
  and the logic of the game and that the datapath
  receives the correct signals to display the correct
  things on the screen.
*/
module control(
  input enable,
  input clock,
  input resetn,

  // USER CONTROL
  input spacebar,
  input enterkey,

  // TO DATAPATH
  output reg [7:0] x,
  output reg [7:0] y,
  output reg [7:0] width,
  output reg [8:0] height,
  output reg [2:0] image_t,
  output game_clock,
  output [7:0] ob_x,
  output [7:0] sp_y,
  output reg [7:0] score,
  
  output obs_out,
  output generate_obs
  );

  assign sp_y = sprite_y;
  assign ob_x = obstacle_x;
  wire increase_score;
  reg [7:0] score_next;
  
  assign obs_out = OBSTACLE_OUT;
  assign generate_obs = GENERATE_OBSTACLE;

  reg [7:0] external_count;
  
  // GAME CLOCK
  wire game_clk;
  assign game_clock = game_clk;

  // // fetch game clk
//	delay dl(
//		.enable(enable),
//		.clk(clock),
//		.resetn(resetn),
//		.game_clk(game_clk)
//		);

  // for testing purposes; clock that slows down by facter of 3.
	delay_counter dcounter(
		.enable(1'b1),
		.clk(clock),
		.resetn(resetn),
		.delay(30'd400000),
		.d_enable(game_clk)
	 );

  // delay counter for score
  delay_counter scoreCounter(
		.enable(game_clk),
		.clk(clock),
		.resetn(resetn),
		.delay(30'd1),
		.d_enable(increase_score)
  );

  always @(posedge increase_score)
  begin
      external_count <= external_count + 1;
		if (current_state != GAME_OVER 
			&& current_state != PLAY_GAME
			&& current_state != INIT_CLOUD
			&& current_state != INIT_GAME
			&& current_state != INIT_BACKGROUND
			&& current_state != INIT_SPRITE
			&& current_state != INIT_OBSTACLE
			&& (external_count % 300 == 0))
			score_next <= score + 1;
		else if (current_state == GAME_OVER)
			score_next <= 8'd0;
  end

  always @(*)
  begin
		if (!resetn)
			score <= 8'd0;
		else
			score <= score_next;
  end

  // GAME STATES
  reg [7:0] current_state, next_state;

  // REGISTER FOR SPRITE POSITION
  reg [7:0] sprite_x, sprite_y;

  // REGISTER FOR CURRENT OBSTACLE position
  reg [7:0] obstacle_x, obstacle_y;
  
  // REGISTER FOR CLOUD POSITION
  reg [7:0] cloud_x, cloud_y;

  // Game state
  localparam INIT_GAME = 8'd0,
             INIT_BACKGROUND = 8'd1,
             INIT_SPRITE = 8'd2,
             INIT_OBSTACLE = 8'd3,
				 INIT_CLOUD = 8'd4,
             PLAY_GAME = 8'd5,
             ERASE_SPRITE = 8'd6,
//             UPDATE_SPRITE_POSITION = 8'd6,
             DRAW_SPRITE = 8'd7,
             ERASE_OBSTACLE = 8'd8,
             UPDATE_OBSTACLE_POSITION = 8'd9,
             DRAW_OBSTACLE = 8'd10,
				 DRAW_CLOUD = 8'd11,
				 UPDATE_CLOUD_POSITION = 8'd12,
				 ERASE_CLOUD = 8'd13,
             GAME_OVER = 8'd14;

  // image dimensions
  localparam IMG_W_BACKGROUND = 8'd160,
             IMG_H_BACKGROUND = 8'd120,
             IMG_W_SPRITE = 8'd16,
             IMG_H_SPRITE = 8'd16,
             IMG_W_OBSTACLE = 8'd16,
             IMG_H_OBSTACLE = 8'd16,
				 IMG_W_CLOUD = 8'd16,
				 IMG_H_CLOUD = 8'd8;

  // image type
  localparam IMG_T_BACKGROUND = 3'b000,
             IMG_T_SPRITE = 3'b001,
             IMG_T_OBSTACLE = 3'b010,
             IMG_T_GAMEOVER = 3'b011,
				 IMG_T_CLOUD = 3'b100;

  // Constants
  localparam PIXEL = 8'd1,
             SPRITE_X_POSITION = 8'd30,
             SPRITE_Y_POSITION = 8'd80,
             SPRITE_TOP_LIM = 8'd40,
             SPRITE_BOTTOM_LIM = 8'd80,
             OBSTACLE_X_POSITION = 8'd160,
             OBSTACLE_Y_POSITION = 8'd80,
				 CLOUD_X_POSITION = 8'd160,
				 CLOUD_Y_POSITION = 8'd100;

  // Edge detection logic
  wire signed [8:0] x_diff, y_diff;
  wire X_OVERLAP, Y_OVERLAP;
  reg HIT;
  assign x_diff = obstacle_x - sprite_x;
  assign y_diff = obstacle_y - sprite_y;

  wire signed [8:0] zero_minus_obstacle_width = 0 - IMG_W_OBSTACLE;
  wire signed [8:0] zero_minus_obstacle_height = 0 - IMG_H_OBSTACLE;
  assign X_OVERLAP = (x_diff >= (zero_minus_obstacle_width) & x_diff <= IMG_W_SPRITE) ? 1'b1 : 1'b0;
  assign Y_OVERLAP = (y_diff >= (zero_minus_obstacle_width) & y_diff <= IMG_H_SPRITE) ? 1'b1 : 1'b0;



  // Flag for determine output signal in state table
  reg SP_LOC;     // sprite location:    0 - sprite on ground; 1 - sprite in air
  reg SP_ORI;     // sprite orientation: 0 - sprite going down;  1 - sprite going up

  localparam IN_AIR = 1'b1,       // SP_LOC
             ON_GRD = 1'b0,
             UP = 1'b1,           // SP_ORI
             DOWN = 1'b0;

  always @(posedge clock)         // update sprite state changes on 50MHz clock
  begin
    if(spacebar) begin            // sprite jumps if spacebar is pressed && on ground
      if(SP_LOC == ON_GRD) begin
        SP_LOC <= IN_AIR;
        SP_ORI <= UP;
      end
    end
	 else if(SP_LOC == IN_AIR) begin
      if(sprite_y >= SPRITE_BOTTOM_LIM && SP_ORI == DOWN) begin         // sprite just landed
        SP_ORI <= DOWN;
        SP_LOC <= ON_GRD;
      end else if(sprite_y <= SPRITE_TOP_LIM && SP_ORI == UP) begin   // sprite just reached top
        SP_ORI <= DOWN;
      end
    end
	 else begin        // initial states
      SP_LOC <= ON_GRD;
      SP_ORI <= DOWN;
    end
  end

  // FSM state table
  always @(*)
  begin: state_table
      case (current_state)
          // GAME INITIALIZATION
          INIT_GAME: next_state = INIT_BACKGROUND;
          INIT_BACKGROUND: next_state = INIT_SPRITE;
          INIT_SPRITE: next_state = INIT_CLOUD;
			 INIT_CLOUD: next_state = INIT_OBSTACLE;
          INIT_OBSTACLE: next_state = PLAY_GAME;

          // spacebar starts game
          PLAY_GAME: next_state = (enterkey) ? ERASE_SPRITE: PLAY_GAME;

          // the cycle begins
          ERASE_SPRITE: next_state = (HIT) ? GAME_OVER: DRAW_SPRITE;
          DRAW_SPRITE: next_state = (HIT) ? GAME_OVER: ERASE_OBSTACLE;

          ERASE_OBSTACLE: next_state = (HIT) ? GAME_OVER: UPDATE_OBSTACLE_POSITION;
			 UPDATE_OBSTACLE_POSITION: next_state = (HIT) ? GAME_OVER: DRAW_OBSTACLE;
          DRAW_OBSTACLE: next_state = (HIT) ? GAME_OVER: ERASE_CLOUD;
			 
			 ERASE_CLOUD: next_state = (HIT) ? GAME_OVER: UPDATE_CLOUD_POSITION;
			 UPDATE_CLOUD_POSITION: next_state = (HIT) ? GAME_OVER: DRAW_CLOUD;
			 DRAW_CLOUD: next_state = (HIT) ? GAME_OVER: ERASE_SPRITE;
			 
          // Game over state
          GAME_OVER: next_state = (enterkey) ? INIT_GAME: GAME_OVER;
      endcase
  end

  // Pseudo random generation of obstacle.
  reg OBSTACLE_OUT;
  reg GENERATE_OBSTACLE;

  localparam RAND_SEED = 30'd4561;
  wire rand_bit;

  random rand(
      .enable(OBSTACLE_OUT),
      .clk(game_clk),
      .resetn(resetn),
      .seed(RAND_SEED),
      .period(16'd600),
      .rand_bit(rand_bit)
    );

  // Output signal corresponding to each state
  always @(posedge game_clk)
  begin: output_signals
    x <= 0;
    y <= 0;
    width <= 0;
    height <= 0;
    image_t <= 0;
    HIT <= X_OVERLAP & Y_OVERLAP;
//	 OBSTACLE_OUT <= (obstacle_x + IMG_W_OBSTACLE < 0) ? 1 : 0;
//	 GENERATE_OBSTACLE <= OBSTACLE_OUT & rand_bit;

    case(current_state)
      INIT_BACKGROUND: begin
        x <= 8'd0;
        y <= 8'd0;
        width <= IMG_W_BACKGROUND;
        height <= IMG_H_BACKGROUND;
        image_t <= IMG_T_BACKGROUND;
      end
      INIT_SPRITE: begin
        // also initializes sprite's x and y coords registers
        sprite_x <= SPRITE_X_POSITION;
        sprite_y <= SPRITE_BOTTOM_LIM;

        x <= SPRITE_X_POSITION;
        y <= SPRITE_BOTTOM_LIM;
        width <= IMG_W_SPRITE;
        height <= IMG_H_SPRITE;
        image_t <= IMG_T_SPRITE;
      end
      INIT_OBSTACLE: begin
        // also initializes sprite's x and y coords registers
        obstacle_x <= OBSTACLE_X_POSITION;
        obstacle_y <= OBSTACLE_Y_POSITION;

        x <= OBSTACLE_X_POSITION;
        y <= OBSTACLE_Y_POSITION;
        width <= IMG_W_OBSTACLE;
        height <= IMG_H_OBSTACLE;
        image_t <= IMG_T_OBSTACLE;
      end
		INIT_CLOUD: begin
		  cloud_x <= CLOUD_X_POSITION;
		  cloud_y <= CLOUD_Y_POSITION;
		  
		  x <= CLOUD_X_POSITION;
		  y <= CLOUD_Y_POSITION;
		  width <= IMG_W_CLOUD;
		  height <= IMG_H_CLOUD;
		  image_t <= IMG_T_CLOUD;
		end
      ERASE_SPRITE: begin
        x <= sprite_x;
        y <= sprite_y;
        width <= IMG_W_SPRITE;
        height <= IMG_H_SPRITE;
        image_t <= IMG_T_BACKGROUND;
      end
      DRAW_SPRITE: begin
        x <= sprite_x;
        y <= sprite_y;
        width <= IMG_W_SPRITE;
        height <= IMG_H_SPRITE;
        image_t <= IMG_T_SPRITE;

	      if(SP_LOC == IN_AIR) begin
          if(SP_ORI == UP)
            sprite_y <= sprite_y - PIXEL;
          else if(SP_ORI == DOWN)
            sprite_y <= sprite_y + PIXEL;
        end
      end
      ERASE_OBSTACLE: begin
        x <= obstacle_x;
        y <= obstacle_y;
        width <= IMG_W_OBSTACLE;
        height <= IMG_H_OBSTACLE;
        image_t <= IMG_T_BACKGROUND;
      end
		UPDATE_OBSTACLE_POSITION: begin
		  obstacle_x <= obstacle_x - PIXEL;
		end
      DRAW_OBSTACLE: begin
        x <= obstacle_x;
        y <= obstacle_y;
        width <= IMG_W_OBSTACLE;
        height <= IMG_H_OBSTACLE;
        image_t <= IMG_T_OBSTACLE;

//        if (!OBSTACLE_OUT)
//		 obstacle_x <= obstacle_x - PIXEL;
			 
//		  if(GENERATE_OBSTACLE) begin
//				obstacle_x <= OBSTACLE_X_POSITION;
//				GENERATE_OBSTACLE <= 0;
//			end
      end
      GAME_OVER: begin
        // reset HIT
        HIT <= 1'b0;
        x <= 8'd0;
        y <= 8'd0;
        width <= IMG_W_BACKGROUND;
        height <= IMG_H_BACKGROUND;
        image_t <= IMG_T_GAMEOVER;
      end
    endcase

  end

// current_state registers
  always@(posedge game_clk)
  begin: state_FFs
      if(!resetn)
          current_state <= INIT_GAME;
      else if(!enable)
          current_state <= INIT_GAME;
      else if(enable)
          current_state <= next_state;
  end


endmodule



/************/
/* DATAPATH */
/************/

/*
  The datapath is responsible for handling the
  inputs that directly go into the VGA.
*/
module datapath(
  input enable,
  input clock,
  input resetn,

  // FROM CONTROL
  input [7:0] x_from_control,
  input [7:0] y_from_control,
  input [7:0] width_from_control,
  input [7:0] height_from_control,
  input [2:0] object_choice,

  // TO VGA
  output reg [7:0] x_out,
  output reg [7:0] y_out,
  output reg [0:0] colour_out
  );

  // TODO: should change all of the colours to 3-bit
  // background x, y, colour
  wire [7:0] bg_x, bg_y;
  wire bg_colour;

  // sprite x, y, colour
  wire [7:0] sp_x, sp_y;
  wire sp_colour;

  // obstacle x, y, colour
  wire [7:0] ob_x, ob_y;
  wire ob_colour;

  // game over x, y, colour
  wire [7:0] go_x, go_y;
  wire go_colour;
  
  //cloud x, y, colour
  wire [7:0] cl_x, cl_y;
  wire cl_colour;

  localparam DRAW_BACKGROUND = 2'd0,
             DRAW_SPRITE = 2'd1,
             DRAW_OBSTACLE = 2'd2,
             DRAW_GAMEOVER = 2'd3,
				 DRAW_CLOUD = 2'd4;

  draw_background d_bg(
    .enable(enable),
    .clk(clock),
    .resetn(resetn),
    .x_in(x_from_control),
    .y_in(y_from_control),
    .width(width_from_control),
    .height(height_from_control),

    .x(bg_x),      // to vga
    .y(bg_y),      // to vga
    .colour(bg_colour) // to vga
    );

  draw_sprite d_sp(
    .enable(enable),
    .clk(clock),
    .resetn(resetn),
    .x_in(x_from_control),
    .y_in(y_from_control),
    .width(8'd16), // width for sprite is hard-coded
    .height(8'd16), // height for sprite is hard-coded

    .x(sp_x),
    .y(sp_y),
    .colour(sp_colour)
    );

 draw_obstacle d_ob(
   .enable(enable),
   .clk(clock),
   .resetn(resetn),
   .x_in(x_from_control),
   .y_in(y_from_control),
   .width(width_from_control),
   .height(height_from_control),

   .x(ob_x),
   .y(ob_y),
   .colour(ob_colour)
   );


 draw_gameover d_go(
   .enable(enable),
   .clk(clock),
   .resetn(resetn),
   .x_in(x_from_control),
   .y_in(y_from_control),
   .width(width_from_control),
   .height(height_from_control),

   .x(go_x),
   .y(go_y),
   .colour(go_colour)
   );
	
  draw_cloud d_cl(
	.enable(enable),
	.clk(clock),
	.resetn(resetn),
	.x_in(x_from_control),
	.y_in(y_from_control),
	.width(width_from_control),
	.height(height_from_control),
	
	.x(cl_x),
	.y(cl_y),
	.colour(cl_colour)
  );

  always @(*)
    begin
      case (object_choice)
        DRAW_BACKGROUND: begin
          x_out = bg_x;
          y_out = bg_y;
          colour_out = bg_colour;
        end
        DRAW_SPRITE: begin
          x_out = sp_x;
          y_out = sp_y;
          colour_out = sp_colour;
        end
        DRAW_OBSTACLE: begin
          x_out = ob_x;
          y_out = ob_y;
          colour_out = ob_colour;
        end
        DRAW_GAMEOVER: begin
          x_out = go_x;
          y_out = go_y;
          colour_out = go_colour;
        end
		  DRAW_CLOUD: begin
			 x_out = cl_x;
			 y_out = cl_y;
			 colour_out = cl_colour;
		  end
        default: begin
          x_out = bg_x;
          y_out = bg_y;
          colour_out = bg_colour;
        end
      endcase
    end

endmodule
