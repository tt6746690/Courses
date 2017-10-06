
#| Unfold

seed, f(seed), f(f(seed)), ..

(repeated f seed n) -> (list seed (f seed) (f (f seed)) ... )

A kind of looping

|#


(steps parallel
       (repeated sqr 2 5))

; • (list 2 (sqr 2) (sqr (sqr 2)) (sqr (sqr (sqr 2))) (sqr (sqr (sqr (sqr 2)))))
; • (list 2 4 (sqr 4) (sqr (sqr 4)) (sqr (sqr (sqr 4))))
; • (list 2 4 16 (sqr 16) (sqr (sqr 16)))
; • (list 2 4 16 256 (sqr 256))
; • (list 2 4 16 256 65536)


(repeated identity "hello" 5)
; (list "hello" "hello" "hello" "hello" "hello")


(repeated rotate-ccw (triangle 10 "solid" "black") 5)
; (list . . . . .)



(define (outline an-image)
  (overlay an-image
           (triangle (* 3 (image-width an-image)) "outline" "darkgreen")))


(repeated rotate-ccw (outline .) 5)

;  View, fix triangle
(map outline (repeated rotate-ccw . 5))


; Functional reactive programming
; model view controller pattern 
(big-bang . ; Model, a state
          [on-tick rotate-ccw 1/2]   ; Controller
          ; unary function called every tick of clock
          [to-draw outline]      ; View
          )

; _view is a function of state_, 
; _controller `add1` called on state for every frame_

; a square bounding box
(image-width (square 20 "solid" "black"))
; 20
(image-width (rotate 45 (square 20 "solid" "black")))
; 28


(rotate 45 .)

; padding space to avoid having to resize size of square box
(define (add-space an-image)
  (overlay an-image
           (circle (/ (square-root (apply + (map sqr (list (image-width an-image)
                                                           (image-height an-image)))))
                      2)
                   "solid" "transparent")))

(define liz (add-space .))

(define (rotate-a-bit an-image)
  (rotate 5 an-image))

(big-bang liz
          [on-tick rotate-a-bit])


(define (draw-liz angle)
  (rotate angle liz))

; State: angle of image
; View: should encompass the image since its not changing
(big-bang 0
          [on-tick add1]        ; controller updates state (i.e. repeated)
          ; <=> (repeated add1 0 100)
          [to-draw draw-liz])   ; view: take current state to generate view
; <=> (map draw-liz (repeated add1 0 100))


; GUI functional programming
; the problem has state (click a button)
; idea is that we dont want to add any state that is not part of the problem
; so result is a pure function of state


(define base-image (add-space (square 20 "solid" (list 6 75 25 25))))
(define base-images (repeated identity base-image 5))
; (list . . . . .)


; map on > 1 list
#;(map f
       (list a b c ...)
       (list x y z ...))
#;(map (f a x) (f b y) (f c z) ...)


(map rotate (list 10 20 30 40 50)
     base-images)

; rotate a list of images
(define (update images)
  (map rotate (repeated add1 1 (length images))
       images))

; combine a list of images to one image
(define (combine images)
  (apply overlay images))

(big-bang base-images
          [on-tick update]
          [to-draw combine])

; Excersice: a model with a list of angles
(define (update2 a-list-of-angles)
  (map + (repeated add1 1 (length a-list-of-angles))
       a-list-of-angles))
(define (draw2 a-list-of-angles)
  (combine (map rotate
                a-list-of-angles
                base-images)))

(define base-angles (list 0 0 0 0 0))
#;(big-bang base-angles
          [on-tick update2]
          [to-draw draw2])
; Exercise: a model thats just one angle
(define (update-list-of-angles a-list-of-angles)
  (map + (repeated add1 1 (length a-list-of-angles))
       a-list-of-angles))

(define base-angle 1)
(define (update3 a-angle)
  (add1 a-angle))
(define (draw3 a-angle)
  (combine (map rotate
                (list-ref (repeated update-list-of-angles base-angles a-angle) (length base-angles))
                base-images)))
(big-bang base-angle
          [on-tick update3]
          [to-draw draw4])


; waiting to do addition if nested recursively
(define (add-up-to n)
  (cond [(positive? n)
         ; This expression is in tail position w.r.t. the whole body
         ; If we get to it, its value is the value of the body
         ; the recursive call is not the last thing that executes (i.e. + op is)
         ; return value of recursive expression is not the value of the body
         ; so have to remember some information
         ; not tail recursive
         (+ n (add-up-to (sub1 n)))]
        [else 0]))

(steps (add-up-to 5))


; Stack builds up during recursion
; add-up-to-n(n) = {n + add-up-to-n(n-1), if n > 0
;                  {0                   , otherwise 
; Space: Theta(n)



(define (add-up-to2 n sum-so-far)
  (cond [(positive? n)
         ; This expression is in tail position w.r.t. the whole body
         ; If we get to it, its value is the value of the body
         ; The recursive call is in tail position w.r.t to the body
         ; the value of expression is the body of expression,
         ; no other operation is being done here
         ; so can throw away the context
         ; Tail recursive (iteration)
         (add-up-to2 (sub1 n) (+ sum-so-far n))]
        [else sum-so-far]))
(steps (add-up-to2 5 0))


; Space: Theta(1)


; Some programming might have implementation that have Theta(n)
; even if in theory (i.e. the second example) its space is Theta(1)

; Recursion is more general then loops
; loops with stacks mimics recursion

; Good idea to have recursion that does not have O(n) stacks
; 






