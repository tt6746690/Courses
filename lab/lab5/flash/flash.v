module flash(HEX0, SW, CLOCK_50);
  input [9:0] SW;
  input CLOCK_50;
  output [6:0] HEX0;

  wire [3:0] number;

  flash_control fc(
      .flash_out(number),
      .choose_speed(SW[1:0]),
      .rClear(SW[9]),
      .dClear(SW[8]),
      .parLoad(SW[7]),
      .rEnable(SW[6]),
      .clock(CLOCK_50)
    );

  decoder d(
      .hex(HEX0),
      .n(number)
    );

endmodule


module flash_control(flash_out, choose_speed, rClear, dClear, parLoad, rEnable, clock);
    // this module utlizes a 4to1 mux which assigns appropriate load_val
    //    for each of the four different speeds, 50MHz, 1Hz, 0.5Hz, and 0.25Hz
    //    Returns a 4 bit hex_display
    input rClear, dClear, parLoad, rEnable, clock;
    input [1:0] choose_speed;
    output [3:0] flash_out;

    reg [27:0] loadVal;
    wire [27:0] RateDivider;

    // 4 to 1 multiplexer selects the countdown values
    always @ (*) begin
      case ( choose_speed )
        2'b00: loadVal = 0;       // 50mHz speed
        2'b01: loadVal = 28'b0010111110101111000010000000;  // 1Hz speed
        2'b10: loadVal = 28'b0101111101011110000100000000;  // 1/2Hz
        2'b11: loadVal = 28'b1011111010111100001000000000;  // 1/4Hz
        default: loadVal = 0;
      endcase
    end


    RateCounter rc(
        .r_out(RateDivider),
        .load_val(loadVal),
        .par_load(parLoad),
        .r_enable(rEnable),
        .r_clock(clock),
        .r_clear(rClear)
      );

    wire dEnable;
    assign dEnable = (RateDivider == 28'b0000000000000000000000000000) ? 1 : 0;

    DisplayCounter dc(
        .d_out(flash_out),
        .d_enable(dEnable),
        .d_clock(clock),
        .d_clear(dClear)
      );
endmodule




module RateCounter(r_out, load_val, par_load, r_enable, r_clock, r_clear);
  //  A parallel-load active-low synchronous reset 28-bit counter that covers
  //    value larger than required 4s period over a 50MHz clock.
  //    Decrements on clock posedge uutputs r_out which is a 28-bit count
  input par_load, r_enable, r_clock, r_clear;
  input [27:0] load_val;
  output reg [27:0] r_out;

  always @(posedge r_clock, negedge r_clear)   // trigger every time clock rises
  begin
    if(r_clear == 1'b0)           // reset when r_clear = 0
      r_out <= 0;
    else if (par_load == 1'b1)    // load loadVal if par_load = 1
      r_out <= load_val;
    else if (r_enable == 1'b1)    // decremenet r_out when r_enable = 1
    begin
      if (r_out == 0)             // reload whenever r_out counts down to zero
        r_out <= load_val;
      else                        // Otherwise decrement r_out
        r_out <= r_out - 1;
    end
  end
endmodule



module DisplayCounter(d_out, d_enable, d_clock, d_clear);
  // Display counter is a 4-bit counter that loops through hexidecimal alphabet
  //  in order, i.e. counters from 4'b0000 to 4'b1111 as output
  input d_enable, d_clock, d_clear;
  output reg [3:0] d_out;

  always @(posedge d_clock, negedge d_clear)
  begin
    if (d_clear == 1'b0)
      d_out <= 4'b0000;
    else if (d_enable == 1'b1)
      d_out <= d_out + 1;
  end
endmodule



module decoder(hex, n);
input [3:0] n;
output [6:0] hex;

  assign hex[0] = ~n[3] & ~n[2] & ~n[1] & n[0] |
                       ~n[3] & n[2] & ~n[1] & ~n[0] |
                       n[3] & n[2] & ~n[1] & n[0] |
                       n[3] & ~n[2] & n[1] & n[0];

  assign hex[1] = n[3] & n[2] & ~n[0] |
                        ~n[3] & n[2] & ~n[1] & n[0] |
                        n[3] & n[1] & n[0] |
                        n[2] & n[1] & ~n[0];

  assign hex[2] = n[3] & n[2] & n[1] |
                        ~n[3] & ~n[2] & n[1] & ~n[0] |
                       n[3] & n[2] & ~n[0];

  assign hex[3] = ~n[3] & n[2] & ~n[1] & ~n[0] |
                       ~n[3] & ~n[2] & ~n[1] & n[0] |
                       n[2] & n[1] & n[0] |
                       n[3] & ~n[2] & n[1] & ~n[0];

  assign hex[4] = ~n[3] & n[0] |
                       ~n[2] & ~n[1] & n[0] |
                       ~n[3] & n[2] & ~n[1];

  assign hex[5] = ~n[3] & ~n[2] & ~n[1] & n[0] |
                       ~n[3] & n[1] & n[0] |
                       n[3] & n[2] & ~n[1] & n[0] |
                       ~n[3] & ~n[2] & n[1];

  assign hex[6] = ~n[3] & ~n[2] & ~n[1] |
                       ~n[3] & n[2] & n[1] & n[0] |
                       n[3] & n[2] & ~n[1] & ~n[0];

endmodule
