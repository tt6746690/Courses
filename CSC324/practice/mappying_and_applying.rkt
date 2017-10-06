#| These are all the questions asking to evaluate expressions involving mapping and applying,
    that have appeared on a CSC104 first test.

 The CSC104 first test is typically in Week 5.

 The current term is the first term that we've had a customized language that can show the
  algebraic reduction rules for map and apply, so for convenience the questions have been
  phrased below in terms of the step form. Except for "★ steps ★", expressions marked with
  "★"s, and the "•" character itself, steps shows exactly what the students need to write out
  for full marks. It's fine if they don't do some or all of it in parallel, or if they don't
  use the convenience of reusing an expression they calculated in an earlier part [expressed
  below with a hide clause].

 If you would like to calibrate, the marks for this kind of question average about 90%, with
  the majority of the lost marks coming from skipping steps. Students find this kind of
  question to be of moderate difficulty relative to other types of first test question,
  and the percentage of marks allocated ends up being a good estimate of the percentage of
  time that they spend on the question. The times are included below, in terms of minutes. |#

#| 3-4 minutes |#

(steps parallel
       (- 20 (apply + (map sqr (list 1 2 3)))))

; (- 20 (apply + (list (sqr 1) (sqr 2) (sqr 3))))
; (- 20 (apply + (list 1 4 9))
; (- 20 (+ 1 4 9))
; (- 20 14)
; (6)



#| 3-4 minutes |#

; A new convenience this term: stepping for aliased functions has been implemented to exclude
;  the step where the variable [in this case f-v] is evaluated to its value [in this case the
;  function flip-vertical]. Some students spontaneously abbreviate, and now we can state simply
;  and precisely the abbreviations they may all use.
(define f-v flip-vertical)
(steps parallel
       (apply above (map f-v (list (triangle 10 "solid" "black")
                                   (triangle 20 "outline" "black")))))

; why triangle evaluated first before anything...
; since list is a function so have to evaluate it first

; WRONG, 
; (apply above (list (f-v (triangle 10 "solid" "black")) (f-v (triangle 20 "outline" "black"))))
; Should be (apply above (map f-v (list . .)))

; (apply above (list (f-v .) (f-v .)))
; (apply above (list . .))
; (above . .)
; .


#| 7 Minutes |#

(define (f n) (- n 7))
(define (g a-list) (= 0 (apply * (map f a-list))))

(steps parallel
       (map f (list 9 8 10)))

; (list (f 9) (f 8) (f 10))
; (list (- 9 7) (- 8 7) (- 10 7))
; (list 2 1 3)


(steps parallel
       [hide (map f (list 9 8 10))]
       (g (list 9 8 10)))

; (= 0 (apply * (map f (list 9 8 10))))
; (= 0 (apply * (list 2 1 3)))
; (= 0 (* 2 1 3))
; (= 0 6)
; #false


; Write a call to ‘g’ that produces #true:



#| 10 Minutes |#

(steps parallel
       (map sqr (range 1 5 1)))

; (map sqr (list 1 2 3 4 5))
; (list (sqr 1) (sqr 2) (sqr 3) (sqr 4) (sqr 5))
; (list 1 4 9 16 25)


(steps parallel
       (apply - (list 12 3)))

; (- 12 3)
; (9)

(define i-h image-height)
(steps parallel
       (map i-h
            (list (beside (square 5 "outline" "black") (square 10 "solid" "black"))
                  (square 7 "outline" "black"))))

; (map i-h (list (beside . .) .))
; (map i-h (list . .))
; (list (i-h .) (i-h .))
; (list 10 7)


#| 14 minutes |#

(steps parallel
       (apply above (list . . .)))

; (above . . .)
;  .


(steps parallel
       (map string? (list "rope" (+ 3 4) "rock")))

; (map string? (list "rope" 7 "rock"))
; (list (string? "rope") (string? 7) (string? "rock"))
; (list #true #false #true)


(define r-ccw rotate-ccw)
(steps parallel
       (map r-ccw (list (beside (triangle 10 "outline" "black") (triangle 10 "solid" "black"))
                        (triangle 10 "outline" "black"))))

; (map r-ccw (list (beside . .) .))
; (map r-ccw (list . .))
; (list (r-ccw ..) (r-ccw .))
; (list . .)

(define s-l string-length)
(define s-a string-append)
(steps parallel
       (apply + (map s-l (list (s-a "rick" "and") "morty"))))

; (apply + (map s-l (list "rickand" "morty")))
; (apply + (list (s-l "rickand") (s-l "morty")))
; (apply + (list 7 5))
; (+ 7 5)
; (12)
