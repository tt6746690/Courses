// SW[0] reset when 0
// SW[1] input signal (w)

// KEY[0] clock signal

// LEDR[2:0] displays current state
// LEDR[9] displays output

module sequence_detector(SW, KEY, LEDR);
    input [9:0] SW;
    input [3:0] KEY;
    output [9:0] LEDR;

    wire w, clock, resetn, out_light;

    reg [2:0] y_Q, Y_D; // y_Q represents current state, Y_D represents next state

    localparam A = 3'b000, B = 3'b001, C = 3'b010, D = 3'b011, E = 3'b100, F = 3'b101, G = 3'b110;

    assign w = SW[1];
    assign clock = ~KEY[0];
    assign resetn = SW[0];

    // State table
    // The state table should only contain the logic for state transitions
    // Do not mix in any output logic.  The output logic should be handled separately.
    // This will make it easier to read, modify and debug the code.

    always @(*)
    begin: state_table
        case (y_Q)
            A: begin
                   if (!w) Y_D = A;
                   else Y_D = B;
               end
            B: begin
                   if(!w) Y_D = A;
                   else Y_D = C;
               end
            C: begin
                   if(!w) Y_D = E;
                   else Y_D = D;
               end
            D: begin
                   if(!w) Y_D = E;
                   else Y_D = F;
               end
            E: begin
                   if(!w) Y_D = A;
                   else Y_D = G;
               end
            F: begin
                   if(!w) Y_D = E;
                   else Y_D = F;
               end
            G: begin
                   if(!w) Y_D = A;
                   else Y_D = C;
               end
            default: Y_D = A;
        endcase
    end // state_table

    // State Register (i.e., FFs)
    always @(posedge clock)
    begin: state_FFs
        if(resetn == 1'b0)
            y_Q <=  A; // Should set reset state to state A
        else
            y_Q <= Y_D;
    end // State Register

    // Output logic
    // Set out_light to 1 to turn on LED when in relevant states
    assign out_light = ((y_Q == F) || (y_Q == G));

    assign LEDR[9] = out_light;
    assign LEDR[2:0] = y_Q;
endmodule
