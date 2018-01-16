#lang racket

; ... is kleene-★, 0 or more of the previous thing
; Here. square brackets are for grouping ★descriptions★
#| def <name>(<parameter-name> [, <parameter-name>] ...):
     <statement>

   <statement> :=
     if <expression>:
       <statement>
     else:
       <statement>
     |  return <expression>

   <expression> :=
       <expression> <op> <expression>
     | <expression>(<expression>, [, <expression>] ...)
     | <variable-name>
     | <number>

   <op> := + | - | <= | >=

   Above is context-free grammar, CFG.
   <statement> is a "non-terminal"
   <expression> ...
   + - <= >= 
|#


#| Some <expression>s:
   0
   x
   x + 0
   x + 0 + 0  (is it (x + 0) + 0 or x + (0 + 0))
   x - y - z  (more than 1 parsing)
   x(0)
   0(0)
|#



; Nanopass compiling
(define (remove-sum c)
  (match c
    [`(sum ,e1 ,op ,e2) `(,op ,e1 ,e2)]
    [_ c]))

; rewrite rule
(define (rewrite rule v)
  (define v′ (rule v))
  (define v′′ (if (list? v′)
                  (map (curry rewrite rule)
                       #; (λ (x) (rewrite rule v))
                       v′)
                  v′))
  (if (equal? v v′′) v (rewrite rule v′′)))

; (rewrite remove-sum code)

