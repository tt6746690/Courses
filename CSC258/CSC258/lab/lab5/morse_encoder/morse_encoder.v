

module morse_encoder(LEDR, SW, KEY, CLOCK_50);
    // A morse encoder that translates letters to morse code
    //    the corresponding dot / dash signal
    input CLOCK_50;
    input [2:0] SW;     // last 8 letter in the alphabet from S to Z
    input [1:0] KEY;    // KEY[1] --  display Morse code for letter specified
                        // KEY[0] --  Asynchronous reset
    output [1:0] LEDR;  // Output used to display Morse code
                        //  -- 0.5s pulse for dots
                        //  -- 1.5s pulse for dashes

    morse_encoder_control me(
        .m_out(LEDR[0]),
        .m_enable(KEY[1]),
        .m_clear(KEY[0]),
        .m_clock(CLOCK_50),
        .m_select(SW[2:0])
      );

endmodule


module morse_encoder_control(m_out, m_enable, m_clear, m_clock, m_select);
    input m_enable, m_clear, m_clock;
    input [2:0] m_select;
    output m_out;

    // Rate divider that sets a 0.5Hz enable
    wire [27:0] RateDivider;

    RateCounter rc(
        .r_out(RateDivider),
        .r_load_val(28'b0001011111010111100001000000),  // 2Hz = 0.5s countdown
        .r_enable(m_enable),
        .r_clock(m_clock),
        .r_clear(m_clear)
      );

    wire s_enable;
    assign s_enable = (RateDivider == 28'b0000000000000000000000000000) ? 1 : 0;

    // look up table
    wire [13:0] pattern;
    LookUpTable lut(
        .lut_out(pattern),
        .lut_select(m_select)
      );

    // 14-bit shift register
    wire [13:0] shifter_state;
    Shifter s(
        .s_out(shifter_state),
        .s_load_val(pattern),
        .shift_left(s_enable),
        .s_clock(m_clock),
        .s_clear(m_clear)
      );
    // first bit is most recent bit shifted to the left edge
    assign m_out = shifter_state[0]; 

endmodule



module Shifter(s_out, s_load_val, shift_left, s_clock, s_clear);
    //  A parallel-load active-low synchronous reset 14-bit shift register
    input shift_left, s_clock, s_clear;
    input [13:0] s_load_val;
    output reg [13:0] s_out;

    reg s_par_load;

    always @(posedge s_clock, negedge s_clear)
    begin
      if(s_clear == 1'b0) begin        // resets when r_clear = 0
        s_out <= 0;
        s_par_load <= 1;               // prepares re parallel load after reset.
      end
      else if (s_par_load == 1'b1) begin    // load s_load_val if s_par_load = 1
        s_out <= s_load_val;
        s_par_load <= 0;                    // just load once.
      end
      else if (shift_left == 1'b1)         // shifts s_load_val to left 1 bit
          s_out <= s_out << 1;
    end
endmodule



module LookUpTable(lut_out, lut_select);
    // A Lookup table that maps 8 letters from S to Z (4-bit) to
    //    a 14 bit representation of corresponding Morse Code
    input [2:0] lut_select;
    output reg [13:0] lut_out;

    always @(*) begin   // A 8 to 1 multiplexer
      case (lut_select)
        3'b000: lut_out = 14'b10_1010_0000_0000;
        3'b001: lut_out = 14'b11_1000_0000_0000;
        3'b010: lut_out = 14'b10_1011_1000_0000;
        3'b011: lut_out = 14'b10_1010_1110_0000;
        3'b100: lut_out = 14'b10_1110_1110_0000;
        3'b101: lut_out = 14'b11_1010_1011_1000;
        3'b110: lut_out = 14'b11_1010_1110_1110;
        3'b111: lut_out = 14'b11_1011_1010_1000;
      endcase
    end
endmodule



module RateCounter(r_out, r_load_val, r_enable, r_clock, r_clear);
  //  A parallel-load active-low synchronous reset 28-bit counter that covers
  //    value larger than required 4s period over a 50MHz clock.
  //    Decrements on clock posedge uutputs r_out which is a 28-bit count
  input r_enable, r_clock, r_clear;
  input [27:0] r_load_val;
  output reg [27:0] r_out;

  reg par_load;

  always @(posedge r_clock, negedge r_clear)   // trigger every time clock rises
  begin
    if(r_clear == 1'b0)           // reset when r_clear = 0
    begin                         // re parallel load after reset.
      r_out <= 0;
      par_load <= 1;
    end
    else if (par_load == 1'b1)    // load load_val if par_load = 1
    begin
      r_out <= r_load_val;
      par_load <= 0;              // just load once.
    end
    else if (r_enable == 1'b1)    // decremenet r_out when r_enable = 1
    begin
      if (r_out == 0)             // reload whenever r_out counts down to zero
        r_out <= r_load_val;
      else                        // Otherwise decrement r_out
        r_out <= r_out - 1;
    end
  end
endmodule
