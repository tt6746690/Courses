#lang racket #| ★ CSC324 2017 Fall Exercise 3 ★ |#

#| The fixed point iteration operation, and some term rewriting. |#
(provide fixed-point deep-map-once If-rule Let-rule)


#| ★ Implement ‘fixed-point’.

 fixed-point : [any → any] any → any

 (fixed-point f v₀) is the first value v in the sequence v₀ (f v₀) (f (f v₀)) ...
  such that v is equal to (f v). |#


; Some test cases.
(module+ test
  (require rackunit)
  
  (define (a v) (min 128 (add1 v)))
  
  (check-equal? (fixed-point a 128) 128)
  (check-equal? (fixed-point a 127) 128)
  (check-equal? (fixed-point a 123) 128)

  (define (b v)
    (cond [(list? v) (first v)]
          [else v]))
  
  (check-equal? (fixed-point b '(((a (b)) c) (d e) f))
                'a)

  ; And some partial partial designs, which you might find useful to fix and work with.
  ; However, you may ignore these if you prefer.
  #;(check-equal? (fixed-point a 123)
                  (local [(define v′ (a 123))]
                    "replace"))
  #;(check-equal? (fixed-point a 127)
                  (local [(define v′ (a 127))]
                    "replace"))
  #;(check-equal? (fixed-point a 128)
                  (local [(define v′ (a 128))]
                    "replace")))


(define (fixed-point f v)
  (cond [(equal? v (f v)) v]
        [else (fixed-point f (f v))]))


#| ★ Implement ‘deep-map-once’.

 deep-map-once : [any → any] any → any

 If (f v) is different from v, then produce (f v).
 If not, and v is a list, produce the list where deep-map-once is used with f on each element of v.
 Otherwise, just produce v. |#


; Example.
(module+ test
  (define (malkovich v)
    (deep-map-once (λ (v) (cond [(list? v) (list* 'malkovich (rest v))]
                                [else v]))
                   v))

  (define v '(the (rain (in spain)) falls (mainly on (the plain))))

  (check-equal? (malkovich v) '(malkovich (rain (in spain)) falls (mainly on (the plain))))

  (check-equal? (malkovich (malkovich v))
                '(malkovich (malkovich (in spain)) falls (malkovich on (the plain))))

  (check-equal? (fixed-point malkovich v)
                '(malkovich (malkovich (malkovich spain)) falls (malkovich on (malkovich plain)))))

(define neq? (λ (a b) (not (equal? a b))))

(define (deep-map-once f v)
  (cond [(neq? (f v) v) (f v)]
        [(list? v) (map (λ (elem) (deep-map-once f elem)) v)]
        [else v]))


#| ★ Implement the Lambda Calculus rewrite rules for ‘If’, and for simple ‘Let’ abbreviations.

 Use ‘match’ appropriately. |#


; A "rewrite" system, which in the context of rewriting source code syntactic shorthands is also
;  called "expansion".
(define (expand f v)
  (fixed-point (λ (v) (deep-map-once f v))
               v))


; Example of a rule and its use with expand.
(module+ test

  (define (Define-rule term)
    (match term

      ; (Define (<f-id> <id>) <e>) → (Define <f-id> (Lambda (<id>) <e>))
      [`(Define (,f-id  ,id)  ,e )  `(Define ,f-id  (Lambda (,id ) ,e ))]
    
      ; Equivalently, in list constructor form for the pattern, and for the result expression.
      #;[(list 'Define (list f-id id) e) (list 'Define f-id (list 'Lambda (list id) e))]

      ; A variable matches anything.
      [t t]))

  (check-equal? (expand Define-rule
                        '(Local [(Define (False t) (Lambda (e) e))
                                 (Define ((True t) e) t)
                                 (Define Not (Lambda (b) (If b False True)))]
                                (Local [(Define (Two f x) (f (f x)))
                                        (Define (g a) (g a))])
                                (g Two)))
                '(Local
                  ((Define False (Lambda (t) (Lambda (e) e)))
                   (Define True (Lambda (t) (Lambda (e) t)))
                   (Define Not (Lambda (b) (If b False True))))
                  (Local ((Define (Two f x) (f (f x)))    ; not expanded since 2 args
                          (Define g (Lambda (a) (g a)))))
                  (g Two))))


; ★ Implement Let-rule to transform terms of the form
#;(Let (<id> <e>)
       <e′>)
;  into
#;((Lambda (<id>) <e′>)
   <e>)
; leaving alone terms that are not of that form.
;
(define (Let-rule term)
  (match term
    [`(Let (,id ,e) ,e′) `((Lambda (,id) ,e′) ,e)]
    [t t]))

(module+ test
  (check-equal? (Let-rule '(Let (g (Lambda (a) (g a))) 
                                (g g)))
                '((Lambda (g) (g g))
                  (Lambda (a) (g a)))))


; ★ Implement If-rule to transform terms of the form
#;(If <c> <t> <e>)
;  into
#;(<c> (Lambda () <t>) (Lambda () <e>))
;  leaving alone terms that are not of that form.
;
(define (If-rule term)
  (match term
    [`(If ,c ,t ,e) `(,c (Lambda () ,t) (Lambda () ,e))]    ; implements short circuiting
    [t t]))

; ★ Make at least one unit test for If-rule, and at least one test with expand on a term
;     that contains nested Ifs.
(module+ test
  (check-equal? (If-rule '(If #t (/ 1 0) 123))
                '(#t (Lambda () (/ 1 0)) (Lambda () 123)))
  (check-equal? (expand If-rule '(If #t (If #t 1 2) (If #f 3 4)))
                '(#t (Lambda ()
                             (#t (Lambda () 1) (Lambda () 2)))
                     (Lambda ()
                             (#f (Lambda () 3) (Lambda () 4))))
                ))


