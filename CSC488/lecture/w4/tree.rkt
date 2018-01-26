#lang slideshow

(provide (rename-out [s-expression→tree-picture tree])
         current-font-size)
(current-font-size 11)

#| Represent an s-expression as a tree.
 
 Compound components are represented by the first element circled,
  with the remaining elements its children.
|#

#| Definitions for the "look" of the tree.
   Helpers separated out from the general tree drawing algorithm. |#
(require slideshow/code)
(define (label token) (inset (code (unsyntax token)) (quotient (current-font-size) 2) 0))
(define (ellipsify picture)
  (cc-superimpose picture (ellipse (+ (quotient (current-font-size) 2) (pict-width picture))
                                   (+ (current-font-size) (pict-height picture)))))
; Combiners.
(define (layout-siblings siblings) (apply ht-append (quotient (current-font-size) 2) siblings))
(define (layout-parent-children parent children)
  (vc-append (current-font-size) parent children))
(define (join! tree parent child)
  (pin-line tree
            parent cb-find
            child ct-find
            #:end-pull 1/10
            #:end-angle (- (/ pi 2))))

(define (s-expression→tree-picture s)
  (cond [(empty? s) (ellipsify (label '||))]
        [(list? s) (let* ([root (ellipsify (label (first s)))]
                          [children (map s-expression→tree-picture (rest s))]
                          [arranged (layout-parent-children
                                     root (layout-siblings children))])
                     (for/fold ([tree arranged])
                       ((child children))
                       (join! tree root child)))]
        [else (label s)]))

(define (read-draw-forever)
  (displayln
   "Please enter an s-expression into the box, or click EOF to end.")
  (define s (read))
  (unless (eof-object? s)
    (print (s-expression→tree-picture s))
    (newline)
    (read-draw-forever)))