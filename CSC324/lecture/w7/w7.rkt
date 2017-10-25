#lang racket


#;(require "mm.rkt")
; 
; (define (Counter)
;   (define count 0)
;   (λ ()
;     (set! count (add1 count))
;     (sub1 count)))

; (define c (Counter))



#; (scale! 20)

; make class
#;(class (Point a b)
    [a a]
    [b b]
    [size (sqrt (+ (sqr a) (sqr b)))]
    [double (set! a (* 2 a))
            (set! b (* 2 b))])

#;(define (Point a b)
    (λ (message)
      (match message
        ['a a]
        ['b b]
        ['size (sqrt (+ (sqr a) (sqr b)))]
        ['double (set! a (* 2 a))
                 (set! b (* 2 b))])))

#;(class (class-id init-id ...)
    [method-id body-expr
               ...]
    ...)

#;(define (class-id init-id ...)
    (λ (message)
      (match message
        ['method-id body-expr
                    ...]
        ...
        )))

#; (define-syntax class   ; at compile time
     (syntax-rules ()
       [(class (class-id init-id ...)
          [method-id body-expr
                     ...]
          ...)
        (define (class-id init-id ...)
          (λ (message)   ; class is a function that accepts a message 
            (match message
              ['method-id body-expr
                          ...]
              ...
              )))]))

; Variadic function
#;((λ v   ; if no parenthesis, arg is variabdic 
     (+ (first v)) (third v))
   1 20 300 4000 50000)



#;(local [(define v (list 1 20 300 4000 50000))]
    (+ (first v) (third v)))

#;((λ message message) 'size) ; message=(list 'size)
#;((λ message message)        ; message='(scale 2)
   'scale 2)

#;(define (Point a b)
    (λ #: Point
      method+arguments
      (match method+arguments
        ['(a) a]
        ['(b) b]
        ['(size) (sqrt (+ (sqr a) (sqr b)))]
        ['(double) (set! a (* 2 a))
                   (set! b (* 2 b))]
        [`(scale ,c) (set! a (* c a))
                     (set! b (* c b))])))
; (define p (Point 8 15))
; (p 'size)
; (p 'scale 10)
; (p 'size)
; (define p1 (Point 8 15))
; (p1 'size)
; (p1 'scale 3)

#; (class (Point a b)
     [(a) a]
     [(b) b]
     [(size) (sqrt (+ (sqr a) (sqr b)))]
     [(double) (set! a (* 2 a))
               (set! b (* 2 b))]
     [(scale c) (set! a (* c a))
                (set! b (* c b))])


; update class syntax
#; (define-syntax class   ; at compile time
     (syntax-rules ()
       [(class (class-id init-id ...)
          (init ...)
          [(method-id parameter-id ...) body-expr
                                        ...]
          ...)
        (define (class-id init-id ...)
          init ...               ; init expressions
          (λ #: class-id
            message+arguments   ; remove paren to make variadic args
            (match message+arguments
              [`(method-id ,parameter-id ...) body-expr
                                              ...]
              ...
              )))]))

#; (class (Point a b)
     [(a) a]
     [(b) b]
     [(size) (sqrt (+ (sqr a) (sqr b)))]
     [(double) (set! a (* 2 a))
               (set! b (* 2 b))]
     [(scale c) (set! a (* c a))
                (set! b (* c b))])


; idea first parens is a list of initialization
; and methods follows after that
#;{
   class (Counter)
    ((define count 0))
    [(increment) (set! count (add1 count))
                 (sub1 count)]
    (define c (Counter))
    (c 'increment)
    (c 'increment)
    (c 'increment)
    }


; implement nested class, where inner class has access to
; variable in the outer scope



(define-syntax class   ; at compile time
  (syntax-rules ()
    [(class (class-id init-id ...)
       (init ...)
       [(method-id parameter-id ...) body-expr
                                     ...]
       ...)
     (define (class-id init-id ...)
       init ...               ; init expressions
       (λ 
           message+arguments   ; remove paren to make variadic args
         (match message+arguments
           [`(method-id ,parameter-id ...) body-expr
                                           ...]
           ...
           )))]))



(class (AddressBook)         ; constructor with no params
  ((define contacts '())     ; constructor with initializer exprs ...
   ; constacts is an instance variable
   (class (AddressBookIterator)   ; nested class
     ((define nextIndex 0))
     [(hasNext) (not (= nextIndex (length contacts)))]     ; contacts in scope of inner class
     [(next) (define next (list-ref contacts nextIndex))
             (set! nextIndex (add1 nextIndex))
             next]))  
  [(addContact name email phone)
   (set! contacts (list* (list name email phone) contacts))]
  [(iterator) (AddressBookIterator)])

; Note iterator instance belongs to 1 addressbook instance
(define ab (AddressBook))
(define i0 (ab 'iterator))
(ab 'addContact 'paul 'pgries '555-123-5678)
(ab 'addContact 'lindsey 'shorser '555-123-5672)

(i0 'next)
(define i1 (ab 'iterator))
(i0 'next)
(i1 'next)



; Control flow
(local []
  (println 'a)
  ; ★
  (println 'b)
  ; ★★
  )

; the 'continuation' of (println 'a) is the point ★ onward
; the 'continuation' of (println 'b) is the point ★★ onward

(+ (+ 1 20)
   ; ★
   (+ 300 4000)
   ; ★★
   )

; the 'continuation' of (+ 1 20) is the point marked with a ★
;   but includes the waiting outer addition.
; The continuation is:
;   add 21 to the result of adding 300 and 4000

; sequentialized version, note eager! 
(local []
  (define r0 (+ 1 20))
  ; ★
  (define r1 (+ 300 4000))
  (+ r0 r1))

; Continuation of (+ 1 20) is something like:
(λ (⊙) (+ ⊙
          (+ 300 4000)))

; Continuation of (+ 300 4000) is something like:
(λ (⊙) (+ 21
          ⊙))


((if (zero? (random 2)) + -) 324)

#; (f a)

((local [] (println "evaluating function expression")
   sqr)
 (local []
   (println "evaluating argument expression")
   18))

; "evaluating function expression"
; "evaluating argument expression"
; ...
; the calls the function


(require (only-in racket/control abort prompt))

; aborts the control flow
#; (local []
     (println 'a)    ; 'a
     (println 'b)    ; 'b
     (abort)
     (println 'c))


; prompt marks its continuation as next global statement (println 'd)
(prompt (local []
          (println 'a)    ; 'a
          (prompt (println 'b)
                  (abort)
                  (println 'c))
          (println 'd)
          ))
(println 'e)
; 'a
; 'b
; 'd
; 'e


; protect second arg from aborting...
(+ (+ 1 20)
   (prompt
    (abort 50000)      ; 
    (+ 300 4000))
   )
; 50021


#; (define (a) (+ 1 (abort 20)))
#; (prompt (a))  ; needs protecting a since abort also abort global print func

(define (a) (prompt (+ 1 (abort 20))))
(+ 300 (a) #;(prompt (a)))


















