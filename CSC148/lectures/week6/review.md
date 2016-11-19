Recursion
=========




This week, we're going to learn about a powerful idea called recursion, which we'll be using in various ways for the rest of the course. However, recursion is much more than just a programming technique: it is really a way thinking about solving problems. Many students when learning recursion for the first time get hung up on tracing through how recursive code works, and completely miss the bigger picture. If you can't think recursively, there's no way you'll be successful in writing recursive code!     

The key idea of recursion is this:    
> identify how an object or problem can be broken down into __smaller__ instances with the __same__ structure.   

In programming, we exploit the recursive structure of objects and problems by making use of recursive functions, which are functions that call themselves in their body.


#### Nested lists  

Consider the problem of computing the sum of a list of numbers. Easy enough:

```python
def sum_list(lst):
    """Return the sum of the items in a list.

    @type lst: list[int]
    @rtype: int
    """
    s = 0
    for num in lst:
        s = s + num
    return s

>>> sum_list([1, 2, 3])
6
```

But what if we make the input structure a bit more complex: a list of lists of numbers? After a bit of thought, we might arrive at using a _nested loop_ to process individual items in the nested list:      

```python
def sum_list2(lst):
    """Return the sum of the items in a list of lists.

    @type lst: list[list[int]]
    @rtype: int
    """
    s = 0
    for list_of_nums in lst:
        for num in list_of_nums:
            s = s + num
    return s

>>> sum_list2([[1], [10, 20], [1, 2, 3]])
37
```

And now what happens if we want yet another layer, and compute the sum of the items in a list of lists of lists of numbers? Some more thought leads to a "nested nested list":

```python
def sum_list3(lst):
    """Return the sum of the items in a list of lists of lists of numbers.

    @type list: list[list[list[int]]]
    @rtype: int
    """
    s = 0
    for list_of_lists_of_nums in lst:
        for list_of_nums in list_of_lists_of_nums:
            for num in list_of_nums:
                s = s + num
    return s

>>> sum_list3([[[1], [10, 20], [1, 2, 3]], [[2, 3], [4, 5]]])
51
```

Of course, you see where this is going: every time we want to add a new layer of nesting to the list, we add a new layer to the for loop. Note that this is quite interesting from a "meta" perspective: the structure of the data is mirrored in the structure of the code which operates on it.

You might have noticed the duplicate code above: in fact, we can use sum_list has a __helper__ for sum_list2, and sum_list2 as a helper for sum_list3:

```python
def sum_list2(lst):
    """Return the sum of the items in a list of lists.

    @type lst: list[list[int]]
    @rtype: int
    """
    s = 0
    for list_of_nums in lst:
        s = s + sum_list(list_of_nms)
    return s

def sum_list3(lst):
    """Return the sum of the items in a list of lists of lists of numbers.

    @type list: list[list[list[int]]]
    @rtype: int
    """
    s = 0
    for list_of_lists_of_nums in lst:
        s = s + sum_list2(list_of_lists_of_nums)
    return s
```


But even this simplification is not the end of the story. If we wanted to implement sum_list10, a function which works on lists with 10 levels of nesting, our only choice would be to first define sum_list4, sum_list5, etc., all the way up to sum_list9.

There is an even bigger problem: no function of this form can handle nested lists with a _non-uniform_ level of nesting among its elements, like  

`[[1, [2]], [[[3]]], 4, [[5, 6], [[[7]]]]]`


We will solve both of these problems at once by first defining a new structure which generalizes this idea of "list of lists of lists of ... of lists of ints". __A nested list__ is one of two things:

1. a single integer (n)
2. or a list of other nested lists ([lst_1, lst_2, ..., lst_n])


The __depth__ of a nested list is the maximum number of times a list is nested inside other lists, with the depth of a single integer being 0. So `[1, 2, 3]` has depth 1, and `[1, [2, [3, 4], 5], [6, 7], 8]` has depth 3.   

This is a __recursive__ definition: it defined nested lists in terms of other nested lists. (Another term for recursive definition is "self-referential definition").

We can use this definition to guide the design of a function which computes the sum on a nested list of numbers:

```python

def nested_sum(obj):
    """Return the sum of the numbers in the nested list <obj>.

    @type obj: int | list
    @rtype: int
    """
    if isinstance(obj, int):
        # obj is an integer
        return obj
    else:
        # obj is a list of nested lists: [lst_1, ..., lst_n]
        s = 0
        for lst_i in obj:
            # each lst_i is a nested list
            s = s + nested_sum(lst_i)
        return ssss
```

This is our first example of a recursive function in Python. Just as we defined a recursive data structure -- __nested lists__ -- we have now defined a recursive function which operates on nested lists. Notice how the structure of the data informs the structure of the code: just as the definition of nested lists separates integers and lists of nested lists into two cases, so too does the function. And as the recursive part of the definition involves a list of nested lists, our code involves a loop over a list, binds lst_i to each inner nested list one at a time, and calls nested_sum on it to compute the sum.

We call the case where obj is an integer the __base case__ of the code: implementing the function's behaviour on this type of input should be very straightforward, and not involve any recursion. The other case, in which obj is a list, is called the __recursive step__: solving the problem in this case requires _decomposing_ the input into smaller nested lists, and calling nested_sum on these individually to solve the problem.

#### Reasoning about recursive calls

We say that the call to nested_sum inside the for loop is a recursive call: it is a call to the same function which is being defined (i.e., the call appears in the body of that function). Such function calls are handled in the same way as all other function calls in Python, but this is actually distracting from the main point.     

When given a function call on a very complex nested list argument, beginners will often attempt to trace through the code carefully, including tracing what happens on each recursive call. Their thoughts go something like "Well, we enter the loop and make a recursive call. That recursive call will cause this other recursive call, and so on." This type of literal tracing is what a computer does, but it's also extremely time-consuming and error-prone.  

Instead, in order to reason about the correctness of recursive code you only need to do two things:   


1. Check that the base case is correct (by manually tracing the code, as you normally would).
2. Check that the recursive step is correct, assuming every recursive call is correct.


It is this assumption which greatly simplifies our reasoning about recursive code, as it means that whenever you trace the code, you don't actually need to "trace into" the recursive calls. For debugging purposes, if your code is incorrect then one of two things is the problem:   

##### The base case is incorrect.     
##### The recursive step is incorrect, even assuming every recursive call is correct.

If you haven't done so already, complete on the recursive tracing worksheet. We have given you a tracing technique involving filling in a table which is much, much easier than manual tracing. We encourage you to use this tracing technique when debugging your own code.

#### Design "Recipe" for recursive functions

Identify the recursive structure of the problem, which can usually be __reduced__ to finding the _recursive structure_ of the input. Figure out if it's a nested list, or some other data type which can be expressed recursively.

Once you do this, you can often write down a code template to guide the structure of your code. For example, the code template for nested lists is:

```python
def f(obj):
    if isinstance(obj, int):
        ...
    else:
        for lst_i in obj:
            ... f(lst_i) ...
```

Identify and implement code for the base case(s). Note that you can usually tell exactly what the base cases are based on the structure of the input. For nested lists, the common base case is when the input is an integer - and if you follow the above template, you won't forget it.   


#### Additional nested list exercises


1. Compute the depth of a nested list, which is the maximum level of nesting in the list. An integer has depth 0.
2. Compute the number of times a number (given as a parameter) occurs within a nested list.
3. Return the a list containing all the odd numbers in a nested list, in the order they appear in the list.
4. Return the a list containing all the odd numbers in a nested list, in the reverse order the appear in the list.
5. Return all the items at a certain depth (given as a parameter) in a nested list.


But in terms of implementation, there is one overriding principle:    

> recursive function calls behave exactly the same as any other function calls. So if you're ever asking a question about how recursion works under the hood, try asking the equivalent question replacing the recursive call with a call to some other helper function, and see if you know the answer.

+ Why do we need a return in the recursive step?

The question is, since the recursive step triggers a recursive call, which will trigger another recursive call, etc., all the way until reaching the base case, then why can't we rely on the base case returning the correct value, and omit any return statements in the recursive step?

```python
def nested_sum(obj):
    if isinstance(obj, int):
        return obj
    else:
        s = 0
        for lst_i in obj:
            s = s + nested_sum(lst_i)
        # omit return?
        s
```



The answer to this question lies not in the behaviour of recursive functions, but in the behaviour of return statements and functions. Consider this (example:

```python
def f():
    return 5

def g():
    f()
```


What happens when we call g?

+ g calls f  
+ f returns 5 to g  
+ g takes the 5 and... stops.  
+ In other words, g returns None, not 5! And this is true regardless of the body of f; it completely depends on the lack of return in g.  

The same is true of recursive functions: without even looking carefully inside the for loop, we can determine that the recursive step will always return None, because it doesn't use return anywhere! So the moral of the story is that any time you want a function or method to return something, you must use the keyword return.


#### Do recursive calls overwrite local variables?

Our code for nested_sum uses local variable s to accumlate the nested sums of each inner nested list. But when each time we make a recursive call, the line s = 0 executes; why doesn't the local variable s get overwritten in each recursive call?

This is one of the fundamental features of functions in almost all programming languages: _every function call has its own namespace of local variables_. In other words, every time you make a function call, it gets its own "set" of local variables, which cannot be influenced by any other function call. This is true when you call two different functions:


```python
def f():
    x = 5

def g():
    x = 100
    f()
    return x  # Returns 100
```


But also when you call the same function, as is the case with recursive functions. So in fact multiple recursive calls never "overwrite" local variables, because each call has its own local variables which are independent from all other calls.

Warning about recursive calls

Finally, we return to the "fundamental assumption" we made when reasoning about recursive code: that every recursive call always works properly. This is a powerful assumption because it greatly simplifies how we can trace our code, but it does come with one caveat.

We can only assume a recursive call is correct when the argument is "smaller" than the original input to the function. What do we mean by smaller? This depends on the type of input, but generally we mean "structurally closer to the base case." For example, in nested lists the base case are the integers, i.e., nested lists of depth 0. In our recursive calls, each lst_i has a smaller depth than the original list, and here "smaller depth" is our indication that these recursive calls are made on smaller inputs.

Here is an example of a bad recursive call, where the depth might actually stay the same between the original input and the input to the recursive call:

```python
def nested_sum(obj):
    if isinstance(obj, int):
        return obj
    else:
        s = 0
        for lst_i in obj:
            s = s + nested_sum([lst_i, 1])    # always a list, recursive function always executed
        return s


>>> nested_sum([1, 2, 3])
RuntimeError: maximum recursion depth exceeded while calling a Python object
We'll talk more about this error later in the course, but for now keep in mind that the "size" of the inputs to recursive calls must always be smaller than the original!

```


So even though the input might look "simpler" than the original, calling this version function results in infinite recursion.


This is one reason we placed such a big emphasis on identifying the recursive structure of the input and problem: if you do so, and respect that structure in your recursive calls, you are almost guaranteed to avoid this problem. This is true for not just nested lists, but also the recursive linked list implementation you saw in Lab 5, and the new recursive data structure we'll see next week.
