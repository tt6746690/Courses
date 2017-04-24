//SW[3:0] data inputs
//HEX0[6:0] output display

module decoder(HEX0, SW);
    input [9:0] SW;
    output [6:0] HEX0;

    assign HEX0[0] = ~SW[3] & ~SW[2] & ~SW[1] & SW[0]  |
                     ~SW[3] & SW[2]  & ~SW[1] & ~SW[0] |
                     SW[3]  & ~SW[2] & SW[1] & SW[0] |
                     SW[3]  & SW[2]  & ~SW[1] & SW[0];

    assign HEX0[1] = SW[3] & SW[1] & SW[0] |
                    SW[2] & SW[1] & ~SW[0] |
                    SW[3] & SW[2] & ~SW[0] |
                    ~SW[3] & SW[2] & ~SW[1] & SW[0];

    assign HEX0[2] = ~SW[3] & ~SW[2] & SW[1] & ~SW[0] |
                   SW[3]  & SW[2]  & ~SW[0] |
                   SW[3]  & SW[2]  & SW[1];

    assign HEX0[3] = ~SW[3] & SW[2] & ~SW[1] & ~SW[0] |
                   ~SW[2] & ~SW[1] & SW[0] |
                   SW[3] & SW[2] & SW[1] & SW[0] |
                   SW[3] & ~SW[2] & SW[1] & ~SW[0];

    assign HEX0[4] = ~SW[3] & SW[0] |
                  ~SW[2] & ~SW[1] & SW[0] |
                  ~SW[3] & SW[2] & ~SW[1] & ~SW[0];

    assign HEX0[5] =  SW[3] & SW[2] & ~SW[1] & SW[0] |
                      ~SW[3] & ~SW[2] & SW[0] |
                      ~SW[3] & ~SW[2] & SW[1] |
                      ~SW[3] & SW[1] & SW[0];

    assign HEX0[6] = ~SW[3] & ~SW[2] & ~SW[1] |
                    SW[3] & SW[2] & ~SW[1] & ~SW[0] |
                    ~SW[3] & SW[2] & SW[1] & SW[0];

endmodule
