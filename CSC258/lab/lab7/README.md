

![](assets/README-5d3b0.png)


__part 1__
+ writing to then reading from RAM

![](assets/README-ba474.png)


__Part 2 VGA__

frame buffer
+ portion of RAM containing a bitmap that is used to refresh a video display from a memory buffer containing .a complete frame of data


__datapath__
+ notice `x_out` and `y_out` loops over a 4x4 coordinate at a predefined coordinate (0,0)

![](assets/README-17d97.png)


__control__
+ controls loading of `ld_x` and `ld_y`

![](assets/README-cff5a.png)


__combination__
+ combination of control unit and datapath, observe (x,y) generated

![](assets/README-15b05.png)



---

__Part 3 move box__

![](assets/README-c8888.png)
+ control sets direction in x and y coordinate upon reaching the boundary of the screen

![](assets/README-1590d.png)
+ delay counter output d_enable does not change for 1000ns


![](assets/README-3e5c1.png)
+ a 4 frame / sec frame counter, notice how output enable high once every 4 clk cycle


![](assets/README-0dae3.png)
+ coordinate counter initialize at position 00001000 with direction = 1,
direction is changed to 0 half way through.
+ notice how coordinate updated once every time enable is high
