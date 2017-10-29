#lang racket



(require (only-in racket/control prompt abort call/comp))


(define-syntax let-continuation
  (syntax-rules ()
    [(let-continuation id
                       expr
                       ...)
     (call/comp (λ (id)
                  expr
                  ...))]))


(define K1 (void))
(define K2 (void))
(prompt (local []
          (println 'a)
          (let-continuation k
                            (set! K1 k)
                            (abort))
          (println 'b)
          (let-continuation k
                            (set! K2 k)
                            (abort))
          (println 'c)))

(println 'z)

; 'a
; 'z
#;(K1)  ; K is a continuation
; 'b
#;(K2)
; 'c




(prompt (+ 1 (local [] (abort 20) 30)))
; 20
(+ 1 (prompt (local [] (abort 20) 30)))
; 21


(define K (void))

; call yield
; records control flow,
;  abort function and give caller a function to get to this point 
(define (yield) (let-continuation k
                                  (abort (λ () (prompt k)))))

(define (h)
  (prompt (local []
            (println 'a)
            (yield)
            (println 'b)
            (yield)
            (println 'c))))


(define a (h))
(define b (a))
(b)



(println "didnt abort the program")

; the caller has control over specifying a behavior at some point
; the caller then callback the behavior later on


(define (show v)
  (println v)
  v)

(local [(define (-< a b c)
          a)]
  (-< (show '♥) (show '◇) (show '♣)))


(local [(define-syntax-rule (-< a b c) a)]
  (-< (show '♥) (show '◇) (show '♣)))
#; (show '♥)



(local [(define-syntax-rule (-< a b c)
          (if (zero? (random 2)) a b))]
  (-< (show '♥) (show '◇) (show '♣)))
#; (show '♥)



(local [(define-syntax-rule (-< a b)
          (if (zero? (random 2)) a b))]
  (-< (-< (show '♥) (show '◇))
      (-< (show '♣) (show '♠))))


(define Q '())
(define-syntax-rule (-< a b)
  (let-continuation k
                    (set! Q (list* (list 'b (λ () (k b)))))
                    a))

(println (* 10 (-< 1
                   2)))

#;(println (* 10  (let-continuation k
                                    (set! Q (λ () (k 2)))
                                    ; insert value 2 to continuation when called
                                    1)))
; 10 when first run
#; (K) ; 20 when called again



(-< (-< 1 2)
    (-< 3 4))
;      1
; (K)  2
; (K)  2
; (K)  2


#; (let-continuation k
                     (set! Q (λ () (k (-< 3 4))))
                     (-< 1 2))


