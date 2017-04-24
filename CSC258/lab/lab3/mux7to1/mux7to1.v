

//SW[6:0] data inputs
//SW[9:7] select signal

//LEDR[0] output display

module mux7to1(LEDR, SW);
    input [9:0] SW;
    output [9:0] LEDR;

    wire [9:0] SW;
    reg [9:0] LEDR;

    always @(*)
    begin
      case (SW[9:7])
        3'b000: LEDR[0] = SW[0];
        3'b001: LEDR[0] = SW[1];
        3'b010: LEDR[0] = SW[2];
        3'b011: LEDR[0] = SW[3];
        3'b100: LEDR[0] = SW[4];
        3'b101: LEDR[0] = SW[5];
        3'b110: LEDR[0] = SW[6];
        default:;
      endcase
    end
endmodule
