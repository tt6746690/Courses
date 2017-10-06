#| CSC104 Test/Exam questions involving recursion on nested lists. |#


#| Question 1 |#

(define (zlub L)
  (cond
    [(number? L) L]
    [else (apply + (map zlub L))]))

; What are the result values of:
#;(zlub 5)
#;(zlub (list 3 4))

; Write out the steps for:
#;(steps parallel
         [hide (zlub 5) (zlub (list 3 4))]
         (zlub (list 3 4 (list 5 6))))
; You may omit showing steps that contain a 'cond' expression.


#| Question 2 |#

; Produces an image containing the text of s, with height 10.
(define (i a-string)
  (string->image a-string 10 "black"))

(define (p s)
  (cond [(string? s) (i s)]
        [else (beside (square 50 "solid" "transparent")
                      (apply above (map p s)))]))

; Recall: 'above' and 'beside' stack/join images aligned by their *centres*.
; Draw the result values of each of the following expressions.
; You may draw the squares as "outline" "black" to help you keep track of positioning.
#;(p "rain")
#;(p (list "in" "spain"))
#;(p (list "on" (list "the" "plain")))
#;(p (list "rain"
           (list "in"
                 "spain")
           "falls"
           (list "on"
                 (list "the"
                       "plain"))))


#| Question 3 |#

(define (f lol)
  (cond [(list? lol) (reverse (apply append (map f lol)))]
        [else (list lol)]))

; What are the result values of:
#;(f 1)
#;(f (list 1 2))

; Write out the steps for:
#;(steps parallel
         [hide (f 1) (f (list 1 2))]
         (f (list (list 1 2) 3 4)))
; You may omit showing steps that contain a 'cond' expression.


#| Question 4 |#

(define (r lol)
  (cond [(list? lol) (reverse (map r lol))]
        [else lol]))

; What are the result values of:
#;(r 1)
#;(r (list 2 3))

; Write out the steps for:
#;(steps parallel
         [hide (r 1) (r (list 2 3))]
         (r (list 1 (list 2 3))))
; You may omit showing steps that contain a 'cond' expression.

; What is the result value of:
#;(r (list (list 1 (list 2 3))))


#| Question 5 |#

(define (e LLL)
  (cond [(list? LLL) (+ 1 (apply maximum (map e LLL)))]
        [else 0]))

; What are the result values of:
#;(e "a")
#;(e (list "b" "c" "d"))

; Write out the steps for:
#;(steps parallel
         [hide (e "a") (e (list "b" "c" "d"))]
         (e (list "a" (list "b" "c" "d"))))
; You may omit showing steps that contain a 'cond' epression.

; What is the result value of:
#;(e (list (list "a"
                 (list "b"
                       "c"))
           (list "d"
                 (list (list "e"
                             "f")
                       "g"))))
; Give a brief explanation, and/or show enough steps to convey why.


#| Question 6 |#

(define (g LLL)
  (cond [(list? LLL) (+ (length LLL)
                        (apply + (map g LLL)))]
        [else 1]))

; What are the result values of:
#;(g "A")
#;(g (list "B" "C" "D"))

; Write out the steps for:
#;(steps parallel
         [hide (g "A") (g (list "B" "C" "D"))]
         (g (list "A" (list "B" "C" "D"))))
; You may omit showing steps that contain a 'cond' epression.
