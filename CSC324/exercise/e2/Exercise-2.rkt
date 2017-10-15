;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "2017-fall-reader.rkt" "csc104")((modname Exercise-2) (compthink-settings #hash((prefix-types? . #f))))
#| ★ CSC324 2017 Fall Exercise 2 ★

 This exercise has you reimplement the functions from Exericse 1, in the point-free
  style that is common in functional programming.

 From the csc104 language, use only the functions that already occur in the sample
  implementations below.

 Do *not*:

   • use recursion, except for using paths in the implementation of paths

   • use any conditionals, except for the main conditional structure in paths

   • define any new non-local functions

   • use any local function definitions of the form:

       (define (<function-name> <parameter-name> ...)
         <body-expression>)

 But you may name functions in point-free style:

   (define <function-name> <function-expression>)

 You may add as many check-expects as you need to debug and check your work.

 Before submitting, you *must* click the "Reindent All" and then "Finer Format" buttons,
  in particular fixing any issues the "Finer Format" button mentions. |#


#| Function specialization functions from lecture and lab. |#

(define (fix-1st f 1st)
  (local [(define (f′ 2nd) 
            (f 1st 2nd))]
    f′))

(define (fix-2nd f 2nd)
  (local [(define (f′ 1st)
            (f 1st 2nd))]
    f′))

; Another one:
(define (fix-1st-2nd f 1st 2nd)
  (local [(define (f′ 3rd)
            (f 1st 2nd 3rd))]
    f′))

#| Function negation function from lecture and lab. |#

(define (¬ p)
  (local [(define (¬p v)
            (not (p v)))]
    ¬p))




; Re-implement include-in-all to use at least one of fix-1st, fix-2nd, ¬, or fix-1st-2nd,
;  following the restrictions stated at the top of this file.
#;(define (include-in-all element list-of-lists)
    (local [(define (include-element one-list)
              (append (list element) one-list))]
      (map include-element list-of-lists)))

(define (include-in-all element list-of-lists)
  (map (fix-1st append (list element)) list-of-lists))



; Re-implement without to use at least one of fix-1st, fix-2nd, ¬, or fix-1st-2nd,
;  following the restrictions stated at the top of this file.
#;(define (without element a-list)
    (local [(define (not-element? e)
              (not (equal? element e)))]
      (filter not-element? a-list)))


; predicate: number? -> bool?
(define (without element a-list)
  (filter (¬ (fix-1st equal? element)) a-list))

; given unchanged
(define (not-in? element a-list)
  (equal? (without element a-list) a-list))

; Re-implement adjacent to use at least one of fix-1st, fix-2nd, ¬, or fix-1st-2nd.
;  following the restrictions stated at the top of this file.
; Also, refactor the body of local to mention shift exactly once.
#;(define (adjacent spot) 
    (local [(define (shift by)
              (map + spot by))]
      (list (shift (list  0 -1))
            (shift (list  0  1))
            (shift (list -1  0))
            (shift (list  1  0)))))

; refactor to mention shift once
#;(define (adjacent spot) 
    (local [(define (shift by)
              (map + spot by))]
      (map shift (list (list 0 -1)
                       (list 0  1)
                       (list -1 0)
                       (list 1  0)))))
 
(define (adjacent spot)
  (map (fix-1st-2nd map + spot) (list (list 0 -1)
                                      (list 0  1)
                                      (list -1 0)
                                      (list 1  0))))


; Re-implement the else clause of the cond in paths to use at least one of fix-1st,
;  fix-2nd, ¬, or fix-1st-2nd, following the restrictions stated at the top of this file.
#;(define (paths to through from)
    (include-in-all
     from
     (cond [(not-in? from through) (list)]
           [(equal? to from) (list (list))]
           [else (local [(define (a-p new-from)
                           (paths to (without from through) new-from))]
                   (apply append (map a-p (adjacent from))))])))


(define (paths to through from)
  (include-in-all
   from
   (cond [(not-in? from through) (list)]
         [(equal? to from) (list (list))]
         [else (apply append (map (fix-1st-2nd paths to (without from through)) (adjacent from)))])))




; Test case.
; include-in-all
(check-expect (include-in-all "hello" (list (list "world" "!")
                                            (list "there" "friend")))
              (list (list "hello" "world" "!")
                    (list "hello" "there" "friend")))

; without
(check-expect (without 2 (list 3 2 4)) (list 3 4))
(check-expect (without 3 (list 3 2 4)) (list 2 4))
(check-expect (without 1 (list 3 2 4)) (list 3 2 4))
(check-expect (without 1 (list 3 1 4 1 5 9 2 6)) (list 3 4 5 9 2 6))
; not-in?
(check-expect (not-in? 2 (list 3 2 4)) #false)
(check-expect (not-in? 3 (list 3 2 4)) #false)
(check-expect (not-in? 1 (list 3 2 4)) #true)
(check-expect (not-in? 1 (list 3 1 4 1 5 9 2 6)) #false)

; adjacent

(check-expect (adjacent (list 0 0)) (list (list  0 -1)
                                          (list  0  1)
                                          (list -1  0)
                                          (list  1  0)))

(define square-of-spots (list (list 0 0) (list 1 0)
                              (list 0 1) (list 1 1)))

(check-expect (paths (list 1 1)
                     square-of-spots
                     (list 0 0))
              (list (list (list 0 0)
                          (list 0 1)
                          (list 1 1))
                    (list (list 0 0)
                          (list 1 0)
                          (list 1 1))))
(define figure-eight (list (list 0 0) (list 1 0) (list 2 0)
                           (list 0 1)            (list 2 1)
                           (list 0 2) (list 1 2) (list 2 2)
                           (list 0 3)            (list 2 3)
                           (list 0 4) (list 1 4) (list 2 4)))
(check-expect (paths (list 2 4) figure-eight (list 0 0))
              (list (list (list 0 0) (list 0 1) (list 0 2) (list 0 3) (list 0 4)
                          (list 1 4) (list 2 4))
                    (list (list 0 0) (list 0 1) (list 0 2)
                          (list 1 2) (list 2 2)
                          (list 2 3) (list 2 4))
                    (list (list 0 0) (list 1 0) (list 2 0)
                          (list 2 1) (list 2 2) (list 2 3) (list 2 4))
                    (list (list 0 0) (list 1 0) (list 2 0)
                          (list 2 1) (list 2 2)
                          (list 1 2) (list 0 2)
                          (list 0 3) (list 0 4)
                          (list 1 4) (list 2 4))))

