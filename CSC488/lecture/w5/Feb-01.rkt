#lang racket
(require "tree.rkt")


#| Let's capture ES6's syntax for the LC.

 x => x
 x(x)
 x(x=>x)

 expr =   id "=>" expr
        | expr "(" expr ")"
        | id

  expr = non-call ["(" expr ")"]*
  non-call = "(" expr ")" | id "=>" expr | id

  x=>x(x)
    (expr (non-call x => (expr (non-call x) "(" (expr (non-call x)) ")")))
  (x=>x)(x)
    (expr (non-call "(" (expr (non-call x => (expr (non-call x)))) ")")
          "(" (expr (non-call x)) ")")
  (x(x))(x)
    
  int
  id = expr
  {expr [[;|NL]+ expr]*} |#

(define (exprs n)
  (cond [(zero? n) (list "x")]
        [else (append* (map (λ (e) (~a "x=>" e))
                           (exprs (- n 1)))
                      (for/list ([n′ n])
                        (for*/list ([e1 (exprs n′)]
                                    [e2 (exprs (- n n′ 1))])
                          (~a e1 "(" e2 ")"))))]))

(require rackunit)
#;#;#;
(check-equal? (exprs 0) (list "x"))
(check-equal? (exprs 1) (list "x=>x" "x(x)"))
(check-equal? (exprs 2) (list "x=>x=>x" "x=>x(x)" "x(x(x))" "x(x)(x)"))
