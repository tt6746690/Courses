#lang racket



#;(let ([x 1]
        [x 2])
    x)
; equivalent
(let ([x 1])
  (let ([x 2])
    x))
(let* ([x 1]
       [x 2])
  2)


; equivalent
(let ([x 1])
  (let ([x (+ x 1)])
    x))
(let* ([x 1]
       [x (+ x 1)])
  x)

; equivalent
(define x 30)
(let ([x x]
      [y x])
  (+ x y))
((λ (x y)
   (+ x y))
 x x)


; letrec: similar to local
#;(letrec ([x (λ () x)]
           [y (+ (x) 500)])
    (+ x y))



#|  Back to Garbage Collection
  State of our program
    a small number of global variables
    larger stack
    larger heap
  Our heap grows, even when we can no longer reference parts of it
  What can be in the global variables
    next : pointer to unused end of heap
    env: pointer to [&En Integer/Boolean/Closure/?], latter from result
    result: Integer/Boolean/&Closure/?
    Look at what can be in result, including after set!

  Want some invariants for what kinds of values can be where

  On stack: push result, return address, environments

  When/why to garbage collect: half of the heap is full
    Factor out allocation
    Currently:
      pair for a closure : (closure <name>)
      pair for environment : (call)
    we'll have not just non-unary environments soon
    Heap: pointer to heap object, along with pointers to code, and non-pointers
      Heap object: n-tuples/records/structs/vectors, each of length at least 1
      Not possible to point into an object
    Make that an invariant at the points when we allocate

    Roots
      Point to objects in the heap.
      Nothing points to them.
      Only way to get to the heap [except for the unused part for allocation].

    Roots are in globals and stack: pointers to environments and closures.
      Need to follow and adjust
      Push the globals onto the stack
        Deal with the stack [and heap]
      Pop the globals
      Loop over the stack:
        if pointer to a closure or environment
          if not there's a forwarding address where the object was/is use that
            copy the closure/environment to new heap
            change the value at the start of that object in the old heap
              to be a poitner to the new location
          adjust the pointer that's in the stack, to be the forwarding address
      Loop over the stuff in new heap:
        do what we just did
|# 




