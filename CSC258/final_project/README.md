# RDR



# resources
- [VGA image display from an UofT ECE lab](http://www.eecg.utoronto.ca/~jayar/ece241_06F/lab7.htm)
- [Good Info on VGA functionalities](http://ee.sharif.edu/~asic/Assignments/Assignment_4/VGA%20Adapter.pdf)
- [pacman tutorial](http://ca.olin.edu/2005/fpga_sprites/results.htm)
- [Keyboard I/O](http://www.johnloomis.org/digitallab/ps2lab1/ps2lab1.html)
- [fantastic read on creating sprites](http://www.academia.edu/24985400/MIFs_Sprites_and_BMP2MIF)
---

# For Prod

* Change the game clock to normal (in FSM)


# Notes


_VGA adaptor_
+ `mif`: memory initialization file
+ resetting the VGA Adapter will not cause the image stored in this file to reappear on
the screen


```bash
convert foo.bmp -colors 24 bar.bmp
./utilities/bmp2mif foo.bmp
```
