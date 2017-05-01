

__PART1: 8 bit counter__


![](assets/README-4dcc5.png)

Logic Utilization and regiter used

![](assets/README-b2c33.png)

maximum clock frequency

![](assets/README-4d9f5.png)

RTL viewer: how quartus synthesized the circuit

![](assets/README-09749.png)

__Part II: Frequency reducer__

![](assets/README-51ee9.png)


Since clock cycle is 20ns / posedge. Then time needed for next flash is 5 X 10^7 X 20 = 1s. We do see a change here

![](assets/README-438aa.png)




__Part III: morse encoder__


+ test rate counter clear and load behave expectedly
test alu look up grabs value

![](assets/README-91564.png)

+ test shift register

![](assets/README-e13bb.png)


+ test that rate divider working properly
+ shift_state remain constant for 1000ns tested

![](assets/README-71af3.png)
