#| CSC 104 Test 2 Recursion Design and Implementation

 These are all the recursions we have ever asked CSC104 students to implement on their second test,
  along with approximate timings based on the weight of the questions. |#


#| 2017 Winter ~20 minutes |#

(define E .)
(check-expect (sandwich .) .)
(check-expect (sandwich .) .)

; Implement sandwich.
; (define sandwich identity)
(define (sandwich bun)
  (local [(define bun-scaled (scale 1/2 bun))]
    (beside (rotate-ccw bun-scaled) E (rotate-cw bun-scaled))))

; Replace "replace" with correct expressions that do not contain any hand-drawn/copy-pasted images.
; Use (branch 0) in the expression for (branch 1), and (branch 1) in the expression for (branch 2).
; Use sandwich where appropriate.
(check-expect (branch 0) .)
(check-expect (branch 0) E)
(check-expect (branch 1) .)
(check-expect (branch 1) (sandwich E))
(check-expect (branch 2) .)
(check-expect (branch 2) (sandwich (sandwich E)))

; Implement branch.
; (define branch identity)
(define (branch depth)
  (cond [(equal? depth 0) E]
        [else (sandwich (branch (sub1 depth)))]))


#| 2016 Winter ~20 minutes |#

(check-expect (r-b .) .)
(check-expect (r-b .) .)

; Implement r-b.
(define (r-b an-image)
  (beside (rotate-ccw an-image) (rotate-cw an-image)))

; Replace "replace" with correct expressions that do not contain any hand-drawn/copy-pasted images.
; Use (tower 0) in the expression for (tower 1), and (tower 1) in the expression for (tower 2).
; Use r-b where appropriate.
(define foo .)
(check-expect (tower 0) .) ; The width of the triangle is 10.
(check-expect (tower 0) foo)
(check-expect (tower 1) .)
(check-expect (tower 1) (above (r-b foo) (scale 2 foo)))
; (check-expect (tower 2) .)
; (check-expect (tower 2) (above (r-b (above (r-b foo) (scale 2 foo))) (scale 2 foo)))

; Implement tower.
; (define (tower depth)
;   (local [(define (stack an-image)
;             (above (r-b an-image) (scale 2 an-image)))]
;     (first (reverse (repeated stack foo (+ depth 1))))))

(define (tower depth)
  (local [(define an-image foo)]
    (cond [(equal? depth 0) an-image]
        [else (above (r-b (tower (sub1 depth))) (scale (* 2 depth) an-image))])))



#| 2016 Fall ~20 minutes |#

(check-expect (double-stack .) .)

; Implement double-stack.
(define (double-stack an-image)
  (above (scale 2 an-image) (beside an-image an-image)))

; Replace "replace" with correct expressions that do not contain any hand-drawn/copy-pasted images.
; Use (pyramid 0) in the expression for (pyramid 1) and (pyramid 1) in the expression for (pyramid 2).
; Use double-stack where appropriate.
(define tri1 .)
(check-expect (pyramid 0) .) ; The width of the triangle is 15.
(check-expect (pyramid 0) tri1)
(check-expect (pyramid 1) .)
(check-expect (pyramid 1) (double-stack tri1))
(check-expect (pyramid 2) .)
(check-expect (pyramid 2) (double-stack (double-stack tri1)))

; Implement pyramid.
(define (pyramid level)
  (cond [(equal? level 0) tri1]
        [else (double-stack (pyramid (sub1 level)))]))


#| 2015 Fall ~15 minutes |#

(define fish .)

; Replace "replace" with correct expressions that do not contain any hand-drawn/copy-pasted images.
; Use (bubble fish 0) in the expression for (bubble fish 1), and (bubble fish 1) in the expression
;  for (bubble fish 2).

(check-expect (bubble . 0) .)
(check-expect (bubble . 0) fish)

(check-expect (bubble . 1) .) ; The circle is twice the height of the fish.
(check-expect (bubble . 1) (beside (rotate 45 fish) (circle (image-height fish) "outline" "black")))

; The new circle is twice the height of ..
(check-expect (bubble . 2) .)


(check-expect (bubble . 3)

              .)

; Implement bubble.
(define (bubble an-image a-number)
  (local [(define (blow-bubble an-image)
            (beside (rotate 45 an-image) (circle (image-height an-image) "outline" "black")))]
    (cond [(equal? a-number 0) an-image]
        [else (blow-bubble (bubble an-image (sub1 a-number)))])))
  


#| 2015 Winter ~15 minutes |#

; Replace "replace" with correct expressions that do not contain any hand-drawn/copy-pasted images.
; Use (f 0) in the expression for (f 1), and (f 2) in the expression for (f 3).
(define T .)
(define S (square 9 "outline" "black"))

; (check-expect (f 0) .)
; (check-expect (f 1) .)
; (check-expect (f 1) (above S (rotate-ccw T )))
; (check-expect (f 2) .)
; (check-expect (f 2) (above (scale 2 S) (rotate-ccw (f 1))))
; (check-expect (f 3) .)
; (check-expect (f 3) (above (scale 3 S) (rotate-ccw (f 2))))



; Implement f.
(define (f level)
  (cond [(equal? 0 level) T]
        [else (above (scale (/ (image-width (f (sub1 level))) (image-width S)) S) (rotate-ccw (f (sub1 level))))]))

; Show the result value for:
#;(f 4)


#| 2014 Fall ~10 minutes |#

; Replace "replace" with correct expressions that do not contain any hand-drawn/copy-pasted images.
; Use (birdy 0) in the expression for (birdy 1), and (birdy 1) in the expression for (birdy 2).

(define a-bird .)
(define bird-size (image-width a-bird))
(define bird-frame (square bird-size "outline" "black"))

(check-expect (birdy 0)  .)
(check-expect (birdy 1) .)
(check-expect (birdy 1) (overlay bird-frame (scale 1/2 a-bird)))
(check-expect (birdy 2) .)
(check-expect (birdy 2) (overlay bird-frame (scale 1/2 (birdy 1))))



; Implement birdy.
(define (birdy level)
  (cond [(equal? level 0) a-bird]
        [else (overlay bird-frame (scale 1/2 (birdy (sub1 level))))]))


#| 2014 Winter ~12 minutes |#

(define another-bird .)

; Replace "replace" with a correct expression that does not contain any hand-drawn/copy-pasted images.

(check-expect (box .) .)

(check-expect (box .) (overlay (square (image-height another-bird) "outline" "black") another-bird))

; Implement box.
; box : image → image
; Produce the image, with a square around it of the same height.
(define (box an-image)
  (overlay (square (image-height an-image) "outline" "black") an-image))


; Replace "replace" with correct expressions that do not contain any hand-drawn/copy-pasted images.
; Use (zone 0) in the expression for (zone 1), and (zone 1) in the expression for (zone 2).
; Use box where appropriate.

(check-expect (zone 0) (circle 25 "outline" "black"))

(check-expect (zone 1) .) ; That circle has radius 25.

(check-expect (zone 1) (rotate 45 (box (zone 0))))

(check-expect (zone 2) .)
(check-expect (zone 2) (rotate 45 (box (zone 1))))

; Implement zone.
(define (zone level)
  (cond [(equal? level 0) (circle 25 "outline" "black")]
        [else (rotate 45 (box (zone (sub1 level))))]))
