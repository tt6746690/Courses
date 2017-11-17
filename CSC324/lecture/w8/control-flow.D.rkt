#lang racket #| Yield |#

#| Early resumable return.

 E.g. Python's and Javascript's generators. |#

(require (only-in racket/control prompt abort call/comp))

(define-syntax-rule (let-continuation id ; Name this whole expression's continuation as ‘id’.
                                      ; Evaluate the body expressions in order.
                                      ; These are *not* part of the continuation.
                                      ; Use the last expression for the value of the whole expression.
                                      body
                                      ...) ; The spot *after* is named ‘id’.
  (call/comp (λ (id) body ...)))

#| Working up to ‘yield’. |#

; 1. Save a continuation to restart a function call “from the outside”.
  
(define K (void))
  
(define (f)
  (prompt (displayln '(f a))
          (let-continuation k
                            (set! K k))
          ; k is here ...
          (displayln '(f b))
          ; ... to here
          )
  (displayln '(f d)))

(f) ; Prints (f a), (f b), (f d).   ; not subsequent exprs in prompt is evaluated after let-continuation
(K) ; Prints (f b).
(K) ; Prints (f b).

; 2. Add the early return behaviour.
  
(define (g)
  (prompt ; coming through here makes abort jump to ★
   (displayln '(g a))
   (let-continuation k
                     (set! K k)
                     (abort) ; jump to ★ and continue
                     )
   ; k is here ...
   (displayln '(g b))
   ; ... to here, i.e. to ★, BUT then return to caller of k.
   )
  ; ★
  (displayln '(g d)))

(g) ; Prints (g a), (g d).
(K) ; Prints (g b).
(K) ; Prints (g b).

; Notice that abort and k represent non-overlapping control flows:
;   • k makes a function out of a slice of the remaining computation,
;      returning to whatever calls it later
;   • abort jumps over that slice

; 3. Try two yields.
  
(define (h)
  (prompt ; coming through here makes abort jump to ★
   (displayln '(h a))
   (let-continuation k1
                     (set! K k1)
                     (abort) ; jump to ★ and continue
                     )
   ; k1 is here ...
   (displayln '(h b))
   (let-continuation k2
                     (set! K k2) 
                     (abort))      ; k1 only captures rest of exprs in prompt,
                                   ; so aborting here is in context of caller, so promp above does not capture it
   ; k2 is here ...
   (displayln '(h c))
   ; ... to here ...,
   ;  except calling k1 is interrupted by the abort causing an early return,
   ;  and calling k2 gets to here and returns to its caller.
   )
  ; ★
  (displayln '(h d)))

(h) ; Prints (h a), (h d).
; Calling ‘K’ prints (h b), but aborts the rest of the program.
#;(K)
; Wouldn't get here:
#;(displayln 'again)
#;(K)
(prompt (K)) ; Prints (h b), the abort is caught by this prompt.
(K) ; Prints (h c).

; 3. Add the prompt protection to the saved continuation.
  
(define (j)
  (prompt (displayln '(j a))
          (let-continuation k
                            (set! K (λ () (prompt (k))))
                            (abort))
          (displayln '(j b))
          (let-continuation k
                            (set! K (λ () (prompt (k))))
                            (abort))
          (displayln '(j c)))
  (displayln '(j d)))

(j)  ; Prints (j a), (h d).
; Let's save that point for later.
(define ja K)
(K)  ; Prints (j b) [and also mutates ‘K’].
(K)  ; Prints (j c).
(ja) ; Prints (j b) [and also mutates ‘K’].
  
; 4. Return the continuation directly to the caller, so that multiple yields,
;     perhaps from different contexts, and multiple resumers, don't conflict.
(define (m)
  (prompt (displayln '(m a))
          (let-continuation k
                            (abort (λ () (prompt (k)))))
          (displayln '(m b))
          (let-continuation k
                            (abort (λ () (prompt (k)))))
          (displayln '(m c))
          '(m d))
  ; Nothing outside the prompt, because we want aborts to come here
  ;  and have their values be used as the return value
  )
 
(define m′ (m)) ; Prints (m a).
(define m′′ (m′)) ; Prints (m b).
(m) ; Prints (m a), and the returned resumer function.
(m′′) ; Prints (m c), and m's return value '(m d).
(m′) ; Prints (m b), and the returned resumer function.
(m′′) ; Prints (m c), and m's return value '(m d).

; returning a lambda, that when invoked, prompts the captured continuation
(define (yield) (let-continuation k (abort (λ () (prompt (k))))))
(define (y)
  (prompt (displayln '(y a))
          (yield)
          (displayln '(y b))
          (yield)
          (displayln '(y c))
          '(y d)))

(define y′ (y)) ; Prints (y a).
(define y′′ (y′)) ; Prints (y b).
(y) ; Prints (y a), and the returned resumer procedure.
(y′′) ; Prints (y c), and y's return value '(y d).
(y′) ; Prints (y b), and the returned resumer procedure.
(y′′) ; Prints (y c), and y's return value '(y d).
