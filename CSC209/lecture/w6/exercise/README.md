
![](assets/README-b5c0b.png)

![](assets/README-4c3ca.png)

![](assets/README-70128.png)

![](assets/README-451aa.png)

![](assets/README-232b1.png)
+ note `make: 'compute_hash' is up to date.` is not printed because `compute_hash` file exists

![](assets/README-db2a7.png)
+ note `make`  traverse to each of `print_ftree.o ftree.o hash_functions.o` and see if any `.c` files are newer, if it is `.o` file is recompiled

![](assets/README-bd226.png)
+ since `hash.h` is dependency of all the `.o` files, they all gets re-compiled if header file is modified

![](assets/README-8a1d1.png)
