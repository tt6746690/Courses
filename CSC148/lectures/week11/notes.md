# Growth of function ?
Growth of function is important.

O:<br>1<br>logn<br>n<br>nlogn<br>n^2<br>2^n<br>n^n  

> efficiency of algorithm matters.    

# Comparison of growth of functions
when n is arbitrarily big, growth of functions highly depends on the dominant term in the function,       

# Time complexity of algorithems
the worst-case time complexity

> an upper bound on the number of operations an algorithm conducts to solve a problem with input size of n.

time complexity is measured in order of number of operations an algorithm used in ...  

**EX1**

```python
def max(list):
    max = list[0]
    for i in range(len(list)):
        if max < list[i]: max = list[i]
    return max
```

exact counting  

```
+ count number of comparisons
```

**EX2**  

```python
def max2(list):
    ma = list[0]
    i = 1
    while i < len(list):
        if max < list[i]: max = list[i]
        i += 1
    return max
```

exact counting (number of comparisons): 2n - 1

therefore time complexity is `O(n)`  

**EX3**  

```python
def silly(n):
    n = 17 * n**(1/2)
    n = n + 3
    if n > 1997:
        print('very bit')
    elif n> 97:
        print('big!')
    else:
        print('not so big')
```

exact counting: 2  

time complexity is `O(1)`


#### Estimating big_O

instead of calculating exact number of operations. and then use the dominant term. Just focus on the dominant part of the algorithm in the first place. The dominant part of the parts are __loops__ and __functino calls__. Hence there are two things to watch

1. need to carefully estimate the number of iterations in the loops in terms of algorithm's input size n  
2. if a called function depends on n ( it has loops that are in terms of n) we should take them into consideration.  

__EX3__  
assume there is not any comparison inside functions print or format.  

__EX1__  
....  
