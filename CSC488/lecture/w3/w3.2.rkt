#lang racket


#; (((λ (x) (λ (y) x))
     488)
    2107)


#; (((λ0 (λ1 1))
     488)
    2107)


(define result (void))
(define result-stack (list))
(define call-stack (list))
(define environment '·)

(define compiled-code #hash{(program . ((closure λ0)
                                        (push-result)  ; function expression put in result stack location
                                        (result 488)   ; argument in result location
                                        (call)         ; → (top-of-result-stack result)
                                        (push-result)  ; put result of ((λ0 (λ1 1)) 488) onto result stack
                                        (result 2107)
                                        (call)))
                            (λ0 . ((closure λ1)))      ; code for lambdas inside.
                            (λ1 . ((variable 1)))})
