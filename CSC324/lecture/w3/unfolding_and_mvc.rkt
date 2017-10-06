#| Unfold, a bit of Functional Reactive Programming, and Mapping across two lists. |#

#| Unfold

 Unfolding in Functional Programming takes a seed value and a function, iterating the function:

  seed, f(seed), f(f(seed)), ...

 In csc104 we capture it with:

   (repeated f seed n)

  whose algebraic reduction is:

   (list seed (f seed) (f (f seed)) ...), of length n. |#

(step parallel
      (repeated sqr 2 5)
      (repeated add1 0 10)
      (repeated identity "hello" 5))

(repeated rotate-ccw . 5)


#| The 'big-bang' Operation |#

#;(big-bang <initial-model>
            [on-tick <controller> <optional-delay>])

; Uncommenting and running the following brings up a window with the initial model image.
;  Every 1/2 second it calls the <controller>, which must be a unary function, on the model
;  to update it, and draws the new model.
#;(big-bang . ; Model
            [on-tick rotate-ccw 1/2]) ; Controller

; In other words, it animates the frames in the list:
#;(repeated . ∞) ; Not actually runnable.
; infinity is not defined
; Except it doesn't actually make a list, and so not an infinite one.


; The big-bang operation is not a function, in particular the on-tick clause is *not*
;  interpreted as a call of 'on-tick'.


; Something "unchanging" that we can do to an image:
(define (outline an-image)
  (overlay an-image
           (triangle (* 3 (image-width an-image)) "outline" "darkgreen")))

; Contrast ...
(map outline (repeated rotate-ccw . 5))
;  ... with:
(repeated rotate-ccw (outline .) 5)


; The post-processing that map did can be animated with a 'to-draw' clause.
#;(big-bang . ; Model
            [on-tick rotate-ccw 1/2] ; Controller
            [to-draw outline]) ; View

; Leaving out a to-draw clause is the same as putting in: [to-draw identity] .


; Most csc104 image functions are not pixel based, instead they maintain knowledge of how
;  the image was built ["structured graphics"]. Here's a function to put enough space
;  around an image, on a circle, so that rotating the image's bounding box doesn't keep
;  enlarging the space. The details aren't important for us, but compare:
(image-width (square 20 "solid" "black"))
(image-width (rotate 45 (square 20 "solid" "black")))

; It's also a good chance to spot a map and apply, although they're probably overkill here.

(define (add-space an-image)
  (overlay an-image
           (circle
            ; Half the length of the diagonal.
            (/ (square-root
                (apply + (map sqr (list (image-width an-image) (image-height an-image)))))
               2)
            "solid" "transparent")))

(define liz (add-space .))

(define (rotate-a-bit an-image)
  (rotate 5 an-image))

#;(big-bang liz
            [on-tick rotate-a-bit])


; The model can be "factored" to remove the constant image: use just a number, for the angle,
;  and then  the view contains the image:

(define (rotated-liz angle)
  (rotate angle liz))

(define (add5 x)
  (+ 5 x))

#;(big-bang 0
            [on-tick add5]
            [to-draw rotated-liz])



; Let's make a slightly more complicated model.

; A 25% opaque greenish square:
(define base-image (add-space (square 20 "solid" (list 0 75 25 25))))
; A list of five of them:
(define base-images (repeated identity base-image 5))

#| Mapping across two lists.

 The map function can take a binary function and two lists of the same length, producing
  a list where the function is called on pairs of corresponding elements from the two lists. |#
#;(map f
       (list a b c ...)
       (list x y z ...))
#;(list (f a x) (f b y) (f c z) ...)

; A function to rotate each of the five squares by a different amount:
(define (update a-list-of-images)
  (map rotate (repeated add1 1 (length a-list-of-images))
       a-list-of-images))

; A function to combine those:
(define (combine a-list-of-images)
  (scale 10 (apply overlay a-list-of-images)))

(map combine (repeated update base-images 5))

#;(big-bang base-images ; Model
            [on-tick update] ; Controller
            [to-draw combine]) ; View

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

(define (update-list-of-angles a-list-of-angles)
  (map + (repeated add1 1 (length a-list-of-angles))
       a-list-of-angles))

(define base-angle 0)
(define (update3 a-angle)
  (add1 a-angle))
(define (draw3 a-angle)
  (combine (map rotate
                (repeated update-list-of-angles base-angles a-angle)
                base-images)))
(big-bang base-angle
          [on-tick update3]
          [to-draw draw3])

; Exercise.
; 1. Recreate is animation, using a list of angles for the model.
; 2. Recreate it using just one angle for the model.
