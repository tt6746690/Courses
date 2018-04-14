
# Conversion bmp -> mif -> ram module

1. Use paint to draw images, outputs 24-bit bmp
2. Convert bmp to mif:  outputs `image.mono.mif` and `image.color.mif`
```sh
make clean && make
./bmp2mif background.bmp     // foo.bmp must be 24-bit bmp
```
3. Use the python script to change bitwidth of `.mif` to 1 bit
```sh
$ python3.6 convert_mif_to_proper.py background.mono.mif background.mif
```
4. Initialize Quartus ram with `(width X height)` words, each holds 1-bit
  + 0 - black
  + 1 - white
