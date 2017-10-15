;; The first three lines of this file were inserted by DrRacket. They record metadata
;; about the language level of this file in a form that our tools can easily process.
#reader(lib "htdp-intermediate-lambda-reader.ss" "lang")((modname html-tags) (read-case-sensitive #t) (teachpacks ()) (htdp-settings #(#t constructor repeating-decimal #f #t none #f () #f)))
#| HTML Tags as Symbols |#

#| This program is written in "Intermediate Student with Lambda", which allows quoting,
    and anonymous functions, but still has the convenient simple testing framework of
    'check-expect'. |#

; Let's start by making a function to get all "sub-parts" of a nested structure that is
;  represented with lists.

(check-expect (sub-parts '(1 (2 3) 4))
              '((1 (2 3) 4)
                1
                (2 3)
                2
                3
                4))

; This design process is essentially the same as for flattening, so I'll take bigger steps,
;  with less commentary, in the design:

(check-expect (sub-parts '(1 (2 3) 4))
              (list* '(1 (2 3) 4)
                     (append '(1)
                             '((2 3) 2 3)
                             '(4))))

(check-expect (sub-parts '(1 (2 3) 4))
              (list* '(1 (2 3) 4)
                     (append (sub-parts 1)
                             (sub-parts '(2 3))
                             (sub-parts 4))))

(check-expect (sub-parts '(1 (2 3) 4))
              (list* '(1 (2 3) 4)
                     (apply append (list (sub-parts 1)
                                         (sub-parts '(2 3))
                                         (sub-parts 4)))))

(check-expect (sub-parts '(1 (2 3) 4))
              (list* '(1 (2 3) 4)
                     (apply append (map sub-parts '(1 (2 3) 4)))))

(define (sub-parts v)
  (list* v
         (cond [(list? v) (apply append (map sub-parts v))]
               [else '()])))


(require 2htdp/batch-io)
(define html (read-xexpr/web "http://www.english.utoronto.ca/about.htm"))

(define html-parts (sub-parts html))

(define (∘ f g)
  (λ (v) (f (g v))))

(define (¬ p)
  (∘ not p))

(define (fix-2nd f 2nd)
  (λ (1st) (f 1st 2nd)))

(define lists (filter list? html-parts))

(define non-empty-lists
  (filter #;(λ (l) (not (empty? l)))
          #;(∘ not empty?)
          (¬ empty?)
          lists))

; All links, i.e. all lists whose first element is 'href:
(filter #;(λ (l) (equal? (first l) 'href))
        (∘ (fix-2nd equal? 'href) first)     ; Note here we use comparison between symbols...
        non-empty-lists)

; From a tag, make a predicate that determines whether a part has that tag.
; make-tag? : symbol → (any → boolean)
(define (make-tag? a-tag)
  (λ (a-part) (and (list? a-part)
                   (not (empty? a-part))
                   (equal? (first a-part) a-tag))))

; The link parts:
(filter (make-tag? 'href) html-parts)
; The image parts:
(filter (make-tag? 'img) html-parts)

; The sources for [only] image parts:
(map second
     (filter (make-tag? 'src)
             (sub-parts (filter (make-tag? 'img) html-parts))))

