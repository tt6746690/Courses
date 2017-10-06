;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "2017-fall-reader.rkt" "csc104")((modname tail) (compthink-settings #hash((prefix-types? . #f))))
; Tail Position, Tail Call, Tail Recursion

(define (add-up-to n)
  
  (cond [(positive? n)
         
         #| The following *expression* is in "tail" *position* w.r.t. the whole body.
            That means: if we get to it, its value is the value of the body.

            The control flow is:
              1. Evaluate 'n', note the number.
              2. Evaluate 'n', note the number.
              3. Call the function referred to by variable 'sub1', passing the number from #2.
              4. Call the function referred to by variable 'add-up-to', passing the result of #3.
              5. Call the function referred to by variable '+', passing the result of #4, and
                  use that call's result as the body's result.
            So the *call* to addition is a tail *call* w.r.t. the evaluating the body. |#
         (+ n
            #| The following expression is not in tail position w.r.t. the whole body.
               In particular, we have to remember some information about what to do on the way
                back/out of the recursion. |#
            (add-up-to (sub1 n)))]
        
        [else 0]))

#| add-up-to(n) = { n + add-up-to(n-1) , if n > 0
                  { 0, otherwise
 That algorithm, whether executed as algebra, or CSC148/207/209 call stack, takes Θ(n) space.

 Note the building of a large arithmetic expression, with the surrounding waiting step #5s, which
  is equivalent to the call stack, and then the additions done on the way back/out. |#
#;(step parallel
        (add-up-to 6))


(define (add-up-to′ n sum-so-far)
  
  (cond [(positive? n)
         
         #| The following *expression* is in tail *position* w.r.t. the whole body.
            The recursive *call* is a tail *call*.
            So this is a tail *recursion*. |#          
         (add-up-to′ (sub1 n) (+ sum-so-far n))]

        [else sum-so-far]))

#| add-up-to′(n, s) = { add-up-to′(n-1, s+n), if n > 0
                      { s, otherwise

 That algorithm is Θ(1) space.

 Note the "accumulation" of the result in the argument, on the way in, with the tail position
  expression literally replacing the body each time, meaning there's literally no record of,
  and nothing to do, on the "way back": |#
#;(step parallel
        (add-up-to′ 6 0))

#| Some implementations of some programming languages are broken, consuming θ(n) space for that.

 This has implications for server code for protocols specified with a finite state machine:
  they can't be implemented in the natural style where each state is a function, and changing
  state is simply calling the corresponding function: the mutual recursion is tail recursion,
  but broken language implementations will run out of memory despite literally requiring only
  a specific [and small!] finite number of states.

 And the language will have to have built-in specialized iteration constructs in order to achieve
  the Θ(1) space expected for "loops", if implementations aren't mandated to preserve the space Θ
  of algorithms.

 We'll return to this in more detail later in the course. |#
