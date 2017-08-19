IO device
+ disk drive with 200 cylinders 0-199. driver starts serving at 100, queue of requests 
    + 23  89  132 42  187
    + head = 100
+ what is total distance traveled to satisfy all requests 
    + FCFS: basically addds the difference between the reqeusts
        + (100-23) + (89-23) + (132-89) + ...
    + SSTF
        +                       23  89  132 42  187
        + distance from 100     77 _11_ 32  58  87
        + distance from 89      66     _43_ 47  98
        + distance from 132     109         90 _55_
        + distance from 187     164        _145_
        + distance from 42     _19_
        + add 11 + 43 + ... + 19


Paging
+ `int X[64][64]` 
+ 4 page frames, each 128 words (`1 int = 1 word`)
+ program manipulate `X` array fits into exactly one pag and always occupy page 0
+ data swappped in and out of ohter three frames
+ `X` is stored in row major order i.e. `X[0][1]` is after `X[0][0]`
+ which will generate lowest number pae faults, compute total number of page faults


```c
// B
for(int j = 0; j <= 63; j++){
    for(int i = 0; i <= 63; i++){
        X[i][j] = 0;
    }
}
```

```c
// A
for(int i = 0; i <= 63; i++){
    for(int j = 0; j <= 63; j++){
        X[i][j] = 0;
    }
}
```


```
pages      0        1       2       3
        program     

Memory 
    [0, 0]          // page 1 start
    [0, 1],         // ...
    ...
    [0, 63],
    [1, 0],
    ...             // ...
    [1, 63],        // page 1 end
    ...
    [63, 63]
```
+ Note since page 128 ints, 
    + so first time on `X[0][0]`, brought in words, `X[0][0]` to `X[1][63]`
+ `A` 
    + `A[0][0]`, `A[1][0]`, ... `A[63][0]`,     (`A[2][0]` would incur another page fault)   
    + `A[0][1]`, `A[1][1]`, ... `A[63][1]`, 
    + ...
    + `A[0][63]`, `A[1][63]`, ... `A[63][63]`
    + implies that 
        + one miss for every 2 reference, 
        + a total of 64 x 64 accesses, 
        + so 32 x 64 total misses
+ `B`
    + `B[0][0]`, `B[0][1]`, ... `B[0][63]`,    
    + `B[1][0]`, `B[1][1]`, ... `B[1][63]`, 
    + ..
    + `B[63][0]`, `B[63][1]`, ... `B[63][63]`,
    + implies 
        + one miss for every 128 reference, 
        + so total of 32 total misses





Unix fs
+ unique index node for each file in system 
    + 8 direct pointer -> pointers to block of size B
    + single indirect pointer -> points to a block containing (B/P) pointers
    + double indirect pointer
        + -> points to a block containtaing (B/P) pointers 
        + -> each (B/P) pointers points to a block containing (B/P) pointers
+ block size B bytes
+ block pointer P bytes
+ what is max file size supported by fs
    + `8B + (B/P)B + (B/P)(B/P)B`




SSL
+ explain SSL