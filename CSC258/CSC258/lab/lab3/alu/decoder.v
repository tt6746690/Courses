//switch[3:0] data inputs
//hex[6:0] output display

module decoder(hex, switch);
    input [3:0] switch;
    output [6:0] hex;

    assign hex[0] = ~switch[3] & ~switch[2] & ~switch[1] & switch[0]  |
                     ~switch[3] & switch[2]  & ~switch[1] & ~switch[0] |
                     switch[3]  & ~switch[2] & switch[1] & switch[0] |
                     switch[3]  & switch[2]  & ~switch[1] & switch[0];

    assign hex[1] = switch[3] & switch[1] & switch[0] |
                    switch[2] & switch[1] & ~switch[0] |
                    switch[3] & switch[2] & ~switch[0] |
                    ~switch[3] & switch[2] & ~switch[1] & switch[0];

    assign hex[2] = ~switch[3] & ~switch[2] & switch[1] & ~switch[0] |
                   switch[3]  & switch[2]  & ~switch[0] |
                   switch[3]  & switch[2]  & switch[1];

    assign hex[3] = ~switch[3] & switch[2] & ~switch[1] & ~switch[0] |
                   ~switch[2] & ~switch[1] & switch[0] |
                   switch[3] & switch[2] & switch[1] & switch[0] |
                   switch[3] & ~switch[2] & switch[1] & ~switch[0];

    assign hex[4] = ~switch[3] & switch[0] |
                  ~switch[2] & ~switch[1] & switch[0] |
                  ~switch[3] & switch[2] & ~switch[1] & ~switch[0];

    assign hex[5] =  switch[3] & switch[2] & ~switch[1] & switch[0] |
                      ~switch[3] & ~switch[2] & switch[0] |
                      ~switch[3] & ~switch[2] & switch[1] |
                      ~switch[3] & switch[1] & switch[0];

    assign hex[6] = ~switch[3] & ~switch[2] & ~switch[1] |
                    switch[3] & switch[2] & ~switch[1] & ~switch[0] |
                    ~switch[3] & switch[2] & switch[1] & switch[0];

endmodule
