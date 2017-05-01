

module mux4to1 (out, in, switch);

input[1:0] switch;
input[3:0] in;
output out;

reg out;
wire[1:0] switch;
wire[3:0] in;

always @(*)
begin
  if(switch == 0)
    out = in[0];
  if(switch == 1)
    out = in[1];
  if(switch == 2)
    out = in[2];
  if(switch == 3)
    out = in[3];
end

endmodule
