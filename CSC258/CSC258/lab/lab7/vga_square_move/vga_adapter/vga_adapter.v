/* VGA Adapter
 * ----------------
 *
 * This is an implementation of a VGA Adapter. The adapter uses VGA mode signalling to initiate
 * a 640x480 resolution mode on a computer monitor, with a refresh rate of approximately 60Hz.
 * It is designed for easy use in an early digital logic design course to facilitate student
 * projects on the Altera DE2 Educational board.
 *
 * This implementation of the VGA adapter can display images of varying colour depth at a resolution of
 * 320x240 or 160x120 superpixels. The concept of superpixels is introduced to reduce the amount of on-chip
 * memory used by the adapter. The following table shows the number of bits of on-chip memory used by
 * the adapter in various resolutions and colour depths.
 * 
 * -------------------------------------------------------------------------------------------------------------------------------
 * Resolution | Mono    | 8 colours | 64 colours | 512 colours | 4096 colours | 32768 colours | 262144 colours | 2097152 colours |
 * -------------------------------------------------------------------------------------------------------------------------------
 * 160x120    |   19200 |     57600 |     115200 |      172800 |       230400 |        288000 |         345600 |          403200 |
 * 320x240    |   78600 |    230400 | ############## Does not fit ############################################################## |
 * -------------------------------------------------------------------------------------------------------------------------------
 *
 * By default the adapter works at the resolution of 320x240 with 8 colours. To set the adapter in any of
 * the other modes, the adapter must be instantiated with specific parameters. These parameters are:
 * - RESOLUTION - a string that should be either "320x240" or "160x120".
 * - MONOCHROME - a string that should be "TRUE" if you only want black and white colours, and "FALSE"
 *                otherwise.
 * - BITS_PER_COLOUR_CHANNEL  - an integer specifying how many bits are available to describe each colour
 *                          (R,G,B). A default value of 1 indicates that 1 bit will be used for red
 *                          channel, 1 for green channel and 1 for blue channel. This allows 8 colours
 *                          to be used.
 * 
 * In addition to the above parameters, a BACKGROUND_IMAGE parameter can be specified. The parameter
 * refers to a memory initilization file (MIF) which contains the initial contents of video memory.
 * By specifying the initial contents of the memory we can force the adapter to initially display an
 * image of our choice. Please note that the image described by the BACKGROUND_IMAGE file will only
 * be valid right after your program the DE2 board. If your circuit draws a single pixel on the screen,
 * the video memory will be altered and screen contents will be changed. In order to restore the background
 * image your circuti will have to redraw the background image pixel by pixel, or you will have to
 * reprogram the DE2 board, thus allowing the video memory to be rewritten.
 *
 * To use the module connect the vga_adapter to your circuit. Your circuit should produce a value for
 * inputs X, Y and plot. When plot is high, at the next positive edge of the input clock the vga_adapter
 * will change the contents of the video memory for the pixel at location (X,Y). At the next redraw
 * cycle the VGA controller will update the contants of the screen by reading the video memory and copying
 * it over to the screen. Since the monitor screen has no memory, the VGA controller has to copy the
 * contents of the video memory to the screen once every 60th of a second to keep the image stable. Thus,
 * the video memory should not be used for other purposes as it may interfere with the operation of the
 * VGA Adapter.
 *
 * As a final note, ensure that the following conditions are met when using this module:
 * 1. You are implementing the the VGA Adapter on the Altera DE2 board. Using another board may change
 *    the amount of memory you can use, the clock generation mechanism, as well as pin assignments required
 *    to properly drive the VGA digital-to-analog converter.
 * 2. Outputs VGA_* should exist in your top level design. They should be assigned pin locations on the
 *    Altera DE2 board as specified by the DE2_pin_assignments.csv file.
 * 3. The input clock must have a frequency of 50 MHz with a 50% duty cycle. On the Altera DE2 board
 *    PIN_N2 is the source for the 50MHz clock.
 *
 * During compilation with Quartus II you may receive the following warnings:
 * - Warning: Variable or input pin "clocken1" is defined but never used
 * - Warning: Pin "VGA_SYNC" stuck at VCC
 * - Warning: Found xx output pins without output pin load capacitance assignment
 * These warnings can be ignored. The first warning is generated, because the software generated
 * memory module contains an input called "clocken1" and it does not drive logic. The second warning
 * indicates that the VGA_SYNC signal is always high. This is intentional. The final warning is
 * generated for the purposes of power analysis. It will persist unless the output pins are assigned
 * output capacitance. Leaving the capacitance values at 0 pf did not affect the operation of the module.
 *
 * If you see any other warnings relating to the vga_adapter, be sure to examine them carefully. They may
 * cause your circuit to malfunction.
 *
 * NOTES/REVISIONS:
 * July 10, 2007 - Modified the original version of the VGA Adapter written by Sam Vafaee in 2006. The module
 *		   now supports 2 different resolutions as well as uses half the memory compared to prior
 *		   implementation. Also, all settings for the module can be specified from the point
 *		   of instantiation, rather than by modifying the source code. (Tomasz S. Czajkowski)
 */

module vga_adapter(
			resetn,
			clock,
			colour,
			x, y, plot,
			/* Signals for the DAC to drive the monitor. */
			VGA_R,
			VGA_G,
			VGA_B,
			VGA_HS,
			VGA_VS,
			VGA_BLANK,
			VGA_SYNC,
			VGA_CLK);
 
	parameter BITS_PER_COLOUR_CHANNEL = 1;
	/* The number of bits per colour channel used to represent the colour of each pixel. A value
	 * of 1 means that Red, Green and Blue colour channels will use 1 bit each to represent the intensity
	 * of the respective colour channel. For BITS_PER_COLOUR_CHANNEL=1, the adapter can display 8 colours.
	 * In general, the adapter is able to use 2^(3*BITS_PER_COLOUR_CHANNEL ) colours. The number of colours is
	 * limited by the screen resolution and the amount of on-chip memory available on the target device.
	 */	
	
	parameter MONOCHROME = "FALSE";
	/* Set this parameter to "TRUE" if you only wish to use black and white colours. Doing so will reduce
	 * the amount of memory you will use by a factor of 3. */
	
	parameter RESOLUTION = "320x240";
	/* Set this parameter to "160x120" or "320x240". It will cause the VGA adapter to draw each dot on
	 * the screen by using a block of 4x4 pixels ("160x120" resolution) or 2x2 pixels ("320x240" resolution).
	 * It effectively reduces the screen resolution to an integer fraction of 640x480. It was necessary
	 * to reduce the resolution for the Video Memory to fit within the on-chip memory limits.
	 */
	
	parameter BACKGROUND_IMAGE = "background.mif";
	/* The initial screen displayed when the circuit is first programmed onto the DE2 board can be
	 * defined useing an MIF file. The file contains the initial colour for each pixel on the screen
	 * and is placed in the Video Memory (VideoMemory module) upon programming. Note that resetting the
	 * VGA Adapter will not cause the Video Memory to revert to the specified image. */


	/*****************************************************************************/
	/* Declare inputs and outputs.                                               */
	/*****************************************************************************/
	input resetn;
	input clock;
	
	/* The colour input can be either 1 bit or 3*BITS_PER_COLOUR_CHANNEL bits wide, depending on
	 * the setting of the MONOCHROME parameter.
	 */
	input [((MONOCHROME == "TRUE") ? (0) : (BITS_PER_COLOUR_CHANNEL*3-1)):0] colour;
	
	/* Specify the number of bits required to represent an (X,Y) coordinate on the screen for
	 * a given resolution.
	 */
	input [((RESOLUTION == "320x240") ? (8) : (7)):0] x; 
	input [((RESOLUTION == "320x240") ? (7) : (6)):0] y;
	
	/* When plot is high then at the next positive edge of the clock the pixel at (x,y) will change to
	 * a new colour, defined by the value of the colour input.
	 */
	input plot;
	
	/* These outputs drive the VGA display. The VGA_CLK is also used to clock the FSM responsible for
	 * controlling the data transferred to the DAC driving the monitor. */
	output [9:0] VGA_R;
	output [9:0] VGA_G;
	output [9:0] VGA_B;
	output VGA_HS;
	output VGA_VS;
	output VGA_BLANK;
	output VGA_SYNC;
	output VGA_CLK;

	/*****************************************************************************/
	/* Declare local signals here.                                               */
	/*****************************************************************************/
	
	wire valid_160x120;
	wire valid_320x240;
	/* Set to 1 if the specified coordinates are in a valid range for a given resolution.*/
	
	wire writeEn;
	/* This is a local signal that allows the Video Memory contents to be changed.
	 * It depends on the screen resolution, the values of X and Y inputs, as well as 
	 * the state of the plot signal.
	 */
	
	wire [((MONOCHROME == "TRUE") ? (0) : (BITS_PER_COLOUR_CHANNEL*3-1)):0] to_ctrl_colour;
	/* Pixel colour read by the VGA controller */
	
	wire [((RESOLUTION == "320x240") ? (16) : (14)):0] user_to_video_memory_addr;
	/* This bus specifies the address in memory the user must write
	 * data to in order for the pixel intended to appear at location (X,Y) to be displayed
	 * at the correct location on the screen.
	 */
	
	wire [((RESOLUTION == "320x240") ? (16) : (14)):0] controller_to_video_memory_addr;
	/* This bus specifies the address in memory the vga controller must read data from
	 * in order to determine the colour of a pixel located at coordinate (X,Y) of the screen.
	 */
	
	wire clock_25;
	/* 25MHz clock generated by dividing the input clock frequency by 2. */
	
	wire vcc, gnd;
	
	/*****************************************************************************/
	/* Instances of modules for the VGA adapter.                                 */
	/*****************************************************************************/	
	assign vcc = 1'b1;
	assign gnd = 1'b0;
	
	vga_address_translator user_input_translator(
					.x(x), .y(y), .mem_address(user_to_video_memory_addr) );
		defparam user_input_translator.RESOLUTION = RESOLUTION;
	/* Convert user coordinates into a memory address. */

	assign valid_160x120 = (({1'b0, x} >= 0) & ({1'b0, x} < 160) & ({1'b0, y} >= 0) & ({1'b0, y} < 120)) & (RESOLUTION == "160x120");
	assign valid_320x240 = (({1'b0, x} >= 0) & ({1'b0, x} < 320) & ({1'b0, y} >= 0) & ({1'b0, y} < 240)) & (RESOLUTION == "320x240");
	assign writeEn = (plot) & (valid_160x120 | valid_320x240);
	/* Allow the user to plot a pixel if and only if the (X,Y) coordinates supplied are in a valid range. */
	
	/* Create video memory. */
	altsyncram	VideoMemory (
				.wren_a (writeEn),
				.wren_b (gnd),
				.clock0 (clock), // write clock
				.clock1 (clock_25), // read clock
				.clocken0 (vcc), // write enable clock
				.clocken1 (vcc), // read enable clock				
				.address_a (user_to_video_memory_addr),
				.address_b (controller_to_video_memory_addr),
				.data_a (colour), // data in
				.q_b (to_ctrl_colour)	// data out
				);
	defparam
		VideoMemory.WIDTH_A = ((MONOCHROME == "FALSE") ? (BITS_PER_COLOUR_CHANNEL*3) : 1),
		VideoMemory.WIDTH_B = ((MONOCHROME == "FALSE") ? (BITS_PER_COLOUR_CHANNEL*3) : 1),
		VideoMemory.INTENDED_DEVICE_FAMILY = "Cyclone II",
		VideoMemory.OPERATION_MODE = "DUAL_PORT",
		VideoMemory.WIDTHAD_A = ((RESOLUTION == "320x240") ? (17) : (15)),
		VideoMemory.NUMWORDS_A = ((RESOLUTION == "320x240") ? (76800) : (19200)),
		VideoMemory.WIDTHAD_B = ((RESOLUTION == "320x240") ? (17) : (15)),
		VideoMemory.NUMWORDS_B = ((RESOLUTION == "320x240") ? (76800) : (19200)),
		VideoMemory.OUTDATA_REG_B = "CLOCK1",
		VideoMemory.ADDRESS_REG_B = "CLOCK1",
		VideoMemory.CLOCK_ENABLE_INPUT_A = "BYPASS",
		VideoMemory.CLOCK_ENABLE_INPUT_B = "BYPASS",
		VideoMemory.CLOCK_ENABLE_OUTPUT_B = "BYPASS",
		VideoMemory.POWER_UP_UNINITIALIZED = "FALSE",
		VideoMemory.INIT_FILE = BACKGROUND_IMAGE;
		
	vga_pll mypll(clock, clock_25);
	/* This module generates a clock with half the frequency of the input clock.
	 * For the VGA adapter to operate correctly the clock signal 'clock' must be
	 * a 50MHz clock. The derived clock, which will then operate at 25MHz, is
	 * required to set the monitor into the 640x480@60Hz display mode (also known as
	 * the VGA mode).
	 */
	
	vga_controller controller(
			.vga_clock(clock_25),
			.resetn(resetn),
			.pixel_colour(to_ctrl_colour),
			.memory_address(controller_to_video_memory_addr), 
			.VGA_R(VGA_R),
			.VGA_G(VGA_G),
			.VGA_B(VGA_B),
			.VGA_HS(VGA_HS),
			.VGA_VS(VGA_VS),
			.VGA_BLANK(VGA_BLANK),
			.VGA_SYNC(VGA_SYNC),
			.VGA_CLK(VGA_CLK)				
		);
		defparam controller.BITS_PER_COLOUR_CHANNEL  = BITS_PER_COLOUR_CHANNEL ;
		defparam controller.MONOCHROME = MONOCHROME;
		defparam controller.RESOLUTION = RESOLUTION;

endmodule
	