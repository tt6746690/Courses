#lang racket #| Compile L1 to sequential language L2 |#

(provide (struct-out compiled:L2) L1→L2)
(module+ test (require rackunit))

#| A2's language L2 is A1's language L2 with three additional expressions. |#

; Jumps to a named statement address, if result is false.
#;(L2: label <name>)      ; Name the current statement location.
#;(L2: jump <name>)       ; Jump/goto a statement location.
#;(L2: jump_false <name>) ; Jump/goto a statement location, when result is false.

#| Compiling L1 to L2 |#

; One additional compilation rule is for L1's new conditional.
#;{(L1: if <n> <e1> <e2> <e3>) →
                               code: {<code-for-e1>
                                      (L2: jump_false else_<n>)
                                      <code-for-e2>
                                      (L2: jump end_<n>)
                                      (L2: label else_<n>)
                                      <code-for-e3>
                                      (L2: label end_<n>)}
                               λs: {<λs-for-e1>
                                    <λs-for-e2>
                                    <λs-for-e3>}}

; A second compilation rule passes through references to an additional function.
#;{(L1: var call/ec) → (L2: closure call_ec)}

#| L1→L2 |#

(struct compiled:L2 (code λs) #:transparent)

; Produce a symbol of the form lambda_<n>.
(require (only-in racket/syntax format-symbol))
(define (lambda_ n) (format-symbol "lambda_~a" n))
(define (else_ n) (format-symbol "else_~a" n))
(define (end_ n) (format-symbol "end_~a" n))


(module+ test
  ; lambda_ and if_
  (check-equal? (lambda_ 1) 'lambda_1)
  (check-equal? (else_ 1) 'else_1)
  (check-equal? (end_ 1) 'end_1)
  ; test L1 producing no lambdas
  (check-equal? (L1→L2 '(L1: var 0))
                (compiled:L2 '((L2: variable 0)) '()))
  (check-equal? (L1→L2 '(L1: datum 1))
                (compiled:L2 '((L2: set_result 1)) '()))
  (check-equal? (L1→L2 '(L1: var +))
                (compiled:L2 '((L2: closure make_add)) '()))
  (check-equal? (L1→L2 '(L1: var *))
                (compiled:L2 '((L2: closure make_multiply)) '()))
  (check-equal? (L1→L2 '(L1: var <))
                (compiled:L2 '((L2: closure make_less_than)) '()))
  (check-equal? (L1→L2 '(L1: var call/ec))
                (compiled:L2 '((L2: closure call_ec)) '()))
  ; test λ
  (check-equal? (L1→L2 '(L1: λ 0 (L1: var 0)))
                (compiled:L2 '((L2: closure lambda_0))
                             '((lambda_0 ((L2: variable 0))))))
  (check-equal? (L1→L2 '(L1: λ 1 (L1: λ 0 (L1: var 1))))
                (compiled:L2 '((L2: closure lambda_1))
                             '((lambda_1 ((L2: closure lambda_0)))
                               (lambda_0 ((L2: variable 1))))))

  ; test set!
  (check-equal? (L1→L2 '(L1: set! 3 (L1: var 2)))
                (compiled:L2 '((L2: variable 2)
                               (L2: set 3))
                             '()))
  ; set! with λ
  (check-equal? (L1→L2 '(L1: λ 1 (L1: set! 0 (L1: λ 0 (L1: var 0)))))
                (compiled:L2 '((L2: closure lambda_1))
                             '((lambda_1 ((L2: closure lambda_0)
                                          (L2: set 0)))
                               (lambda_0 ((L2: variable 0))))))
  ; test app
  (check-equal? (L1→L2 '(L1: app (L1: λ 0 (L1: var 0)) (L1: datum 2)))
                (compiled:L2 '((L2: closure lambda_0)
                               (L2: push_result)
                               (L2: set_result 2)
                               (L2: call))
                             '((lambda_0 ((L2: variable 0))))))

  ; test if
  (check-equal? (L1→L2 '(L1: if 12 (L1: datum 1) (L1: datum 2) (L1: datum 3)))
                (compiled:L2 '((L2: set_result 1)
                               (L2: jump_false else_12)
                               (L2: set_result 2)
                               (L2: jump end_12)
                               (L2: label else_12)
                               (L2: set_result 3)
                               (L2: label end_12))
                             '()))
  )

(define (L1→L2 e)
  (match e
    ; λ
    [`(L1: λ ,<n> ,<e>)
     (define lambda-name (lambda_ <n>))
     (match-define (compiled:L2 <e>-code <e>-λs) (L1→L2 <e>))
     (compiled:L2 `((L2: closure ,lambda-name))
                  (append `((,lambda-name ,<e>-code))
                          <e>-λs))]
    ; app
    [`(L1: app ,<e1> ,<e2>)
     (match-define (compiled:L2 <e1>-code <e1>-λs) (L1→L2 <e1>))
     (match-define (compiled:L2 <e2>-code <e2>-λs) (L1→L2 <e2>))
     (compiled:L2 (append <e1>-code
                          '((L2: push_result))
                          <e2>-code
                          '((L2: call)))
                  (append <e1>-λs <e2>-λs))
     ]
    ; set!
    [`(L1: set! ,<n> ,<e>)
     (match-define (compiled:L2 <e>-code <e>-λs) (L1→L2 <e>))
     (compiled:L2 (append <e>-code
                          `((L2: set ,<n>)))
                  <e>-λs)]
    
    ; if
    [`(L1: if ,<n> ,<e1> ,<e2> ,<e3>)
     (define else_name (else_ <n>))
     (define end_name (end_ <n>))
     (match-define (compiled:L2 <e1>-code <e1>-λs) (L1→L2 <e1>))
     (match-define (compiled:L2 <e2>-code <e2>-λs) (L1→L2 <e2>))
     (match-define (compiled:L2 <e3>-code <e3>-λs) (L1→L2 <e3>))
     (compiled:L2 (append <e1>-code
                          `((L2: jump_false ,else_name))
                          <e2>-code
                          `((L2: jump ,end_name)
                            (L2: label ,else_name))
                          <e3>-code
                          `((L2: label ,end_name)))
                  (append <e1>-λs <e2>-λs <e3>-λs))]
    ; L1 statement that produce no lambdas
    [`(L1: var +) (compiled:L2 '((L2: closure make_add)) '())]
    [`(L1: var *) (compiled:L2 '((L2: closure make_multiply)) '())]
    [`(L1: var <) (compiled:L2 '((L2: closure make_less_than)) '())]
    [`(L1: var call/ec) (compiled:L2 '((L2: closure call_ec)) '())]
    [`(L1: var ,<n>) (compiled:L2 `((L2: variable ,<n>)) `())]
    [`(L1: datum ,<i>)
     (compiled:L2 `((L2: set_result ,<i>)) '())]
    [_ (compiled:L2 '() '())]))
