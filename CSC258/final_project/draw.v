//// For testing:
//`include "rams/background_ram.v";
//`include "rams/sprite_ram.v";
//`include "rams/obstacle_ram.v";
//`include "rams/gameover_ram.v";

module draw_background(       // 120x160 dimension using 19200x1 ram
  input enable,
  input clk,
  input resetn,
  input [7:0] x_in,           // upper left coordinate (x_in, y_in)
  input [7:0] y_in,
  input [7:0] width,          // dimension of rectangle to be drawn (width, height)
  input [7:0] height,
  output [7:0] x,              // outputs (x, y, colour) every CLOCK_50
  output [7:0] y,
  output colour
  );

  localparam MEM_BITWIDTH = 160;

  reg [7:0] addr;
  reg [7:0] i;

  // Loops over x \in [x_in, x_in + width); y \in (y_in, y_in + height)
  always @(posedge clk)
  begin
    if(!resetn) begin
      i <= x_in;
      addr <= y_in;
    end
    else if(enable) begin
      if (i < x_in + width - 1)
        i <= i + 1;
      else begin
        if (addr < y_in + height - 1)
          addr <= addr + 1;
        else
          addr <= y_in;
        i <= x_in;
      end
    end
    else begin
      i <= x_in;
      addr <= y_in;
    end
  end

  // fetch colour bit from background memory
  wire [14:0] addr_map;
  assign addr_map = MEM_BITWIDTH * addr + i;

  wire mem;

  background_ram bg_ram(        // 19200x1 ram
    .address(addr_map),
    .clock(clk),
    .data(1'b0),
    .wren(1'b0),                   // Read only
    .q(mem)                     // A single 256-bit row
    );

  assign x = i;
  assign y = addr;
  assign colour = mem;

endmodule



module draw_gameover(       // 120x160 dimension using 19200x1 ram
  input enable,
  input clk,
  input resetn,
  input [7:0] x_in,           // upper left coordinate (x_in, y_in)
  input [7:0] y_in,
  input [7:0] width,          // dimension of rectangle to be drawn (width, height)
  input [7:0] height,
  output [7:0] x,              // outputs (x, y, colour) every CLOCK_50
  output [7:0] y,
  output colour
  );

  localparam MEM_BITWIDTH = 160;

  reg [7:0] addr;
  reg [7:0] i;

  // Loops over x \in [x_in, x_in + width); y \in (y_in, y_in + height)
  always @(posedge clk)
  begin
    if(!resetn) begin
      i <= x_in;
      addr <= y_in;
    end
    else if(enable) begin
      if (i < x_in + width - 1)
        i <= i + 1;
      else begin
        if (addr < y_in + height - 1)
          addr <= addr + 1;
        else
          addr <= y_in;
        i <= x_in;
      end
    end
    else begin
      i <= x_in;
      addr <= y_in;
    end
  end

  // fetch colour bit from background memory
  wire [14:0] addr_map;
  assign addr_map = MEM_BITWIDTH * addr + i;

  wire mem;

  gameover_ram go_ram(        // 19200x1 ram
    .address(addr_map),
    .clock(clk),
    .data(1'b0),
    .wren(1'b0),                   // Read only
    .q(mem)                     // A single 256-bit row
    );

  assign x = i;
  assign y = addr;
  assign colour = mem;

endmodule


module draw_sprite(           // 16x16 dimension using 256x1 ram
  input enable,
  input clk,
  input resetn,
  input [7:0] x_in,           // upper left coordinate (x_in, y_in)
  input [7:0] y_in,
  input [7:0] width,          // dimension of rectangle to be drawn (width, height)
  input [7:0] height,
  output [7:0] x,             // outputs (x, y, colour) every CLOCK_50
  output [7:0] y,
  output colour
  );

  localparam MEM_BITWIDTH = 16;    // 16

  reg [7:0] addr;
  reg [7:0] i;

  // Loops over x \in (0, width); y \in (0, height)
  always @(posedge clk)
  begin
    if(!resetn) begin
      i <= 0;
      addr <= 0;
    end
    else if(enable) begin
      if (i < width - 1)
        i <= i + 1;
      else begin
        if (addr < height - 1)
          addr <= addr + 1;
        else
          addr <= 0;
        i <= 0;
      end
    end
    else begin
      i <= 0;
      addr <= 0;
    end
  end

  // fetch colour bit from sprite memory
  wire [7:0] addr_map;          // 256x1 ram
  assign addr_map = MEM_BITWIDTH * addr + i;

  wire mem;

  sprite_ram sp_ram(        // 19200x1 ram
    .address(addr_map),
    .clock(clk),
    .data(1'b0),
    .wren(1'b0),                   // Read only
    .q(mem)                     // A single 256-bit row
    );

  assign x = x_in + i;
  assign y = y_in + addr;
  assign colour = mem;

endmodule


module draw_obstacle(           // 16x16 dimension using 256x1 ram
 input enable,
 input clk,
 input resetn,
 input [7:0] x_in,           // upper left coordinate (x_in, y_in)
 input [7:0] y_in,
 input [7:0] width,          // dimension of rectangle to be drawn (width, height)
 input [7:0] height,
 output [7:0] x,             // outputs (x, y, colour) every CLOCK_50
 output [7:0] y,
 output colour
 );

 localparam MEM_BITWIDTH = 16;    // 16

 reg [7:0] addr;
 reg [7:0] i;

 // Loops over x \in (0, width); y \in (0, height)
 always @(posedge clk)
 begin
   if(!resetn) begin
     i <= 0;
     addr <= 0;
   end
   else if(enable) begin
     if (i < width - 1)
       i <= i + 1;
     else begin
       if (addr < height - 1)
         addr <= addr + 1;
       else
         addr <= 0;
       i <= 0;
     end
   end
   else begin
     i <= 0;
     addr <= 0;
   end
 end

 // fetch colour bit from sprite memory
 wire [7:0] addr_map;          // 256x1 ram
 assign addr_map = MEM_BITWIDTH * addr + i;

 wire mem;

 obstacle_ram ob_ram(        // 256x1 ram
   .address(addr_map),
   .clock(clk),
   .data(1'b0),
   .wren(1'b0),                   // Read only
   .q(mem)                     // A single 256-bit row
   );

 assign x = x_in + i;
 assign y = y_in + addr;
 assign colour = mem;

endmodule

module draw_cloud(           // 16x16 dimension using 256x1 ram
 input enable,
 input clk,
 input resetn,
 input [7:0] x_in,           // upper left coordinate (x_in, y_in)
 input [7:0] y_in,
 input [7:0] width,          // dimension of rectangle to be drawn (width, height)
 input [7:0] height,
 output [7:0] x,             // outputs (x, y, colour) every CLOCK_50
 output [7:0] y,
 output colour
 );

 localparam MEM_BITWIDTH = 16;    // 16

 reg [7:0] addr;
 reg [7:0] i;

 // Loops over x \in (0, width); y \in (0, height)
 always @(posedge clk)
 begin
   if(!resetn) begin
     i <= 0;
     addr <= 0;
   end
   else if(enable) begin
     if (i < width - 1)
       i <= i + 1;
     else begin
       if (addr < height - 1)
         addr <= addr + 1;
       else
         addr <= 0;
       i <= 0;
     end
   end
   else begin
     i <= 0;
     addr <= 0;
   end
 end

 // fetch colour bit from sprite memory
 wire [7:0] addr_map;          // 256x1 ram
 assign addr_map = MEM_BITWIDTH * addr + i;

 wire mem;

 cloud_ram cl_ram(        // 256x1 ram
   .address(addr_map),
   .clock(clk),
   .data(1'b0),
   .wren(1'b0),                   // Read only
   .q(mem)                     // A single 256-bit row
   );

 assign x = x_in + i;
 assign y = y_in + addr;
 assign colour = mem;

endmodule
