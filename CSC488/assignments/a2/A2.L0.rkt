#lang racket #| Macro System for Languages that Expand to L0 |#

#| There is nothing to implement in this file, but you need to understand it to implement A2.M0.rkt.

 Consider adding error messages to the expand function, to help you during development, and to make
  sure you understand that function. You could also add debugging statements to observe expansion. |#

(provide (struct-out transformer)
         define-transformer
         expand)

#| Macros
   ======
 A macro in this system is a pair of:
   • a name
   • a function to transform expressions of the form (name <part> ...)

 The struct ‘transformer’ bundles those two together.

 The form (define-transformer id clause ...) is a convenience, used in A2.M0.rkt, to define
  transformers based on match clauses. |#

#| Expansion to L0
   ===============
 The function expand takes an expression to expand, and a list of transformers, and expands
  the expression to an L0 expression according to the transformers and knowledge of which
  parts of L0 expressions are themselves L0 expressions.

 The macro system assumes the language being transformed expects integer, variable, and unnamed
  compound expressions. If a symbol is encountered as an expression, it's wrapped as (*id* <id>)
  so that a corresponding transformer can give it meaning. Similarly, integers are wrapped as
  (*datum* <integer>), and a list not transformed by a transformer nor one of the L0 forms is
  wrapped as (*app* <e> ...). The list of transformers must include transformers for *id*, *datum*,
  or *app* accordingly, otherwise expansion will keep wrapping those with *app* [since they're
  lists without a transformer]. |#

(module+ test (require rackunit))

; Struct for a macro: the name of the form it expands, and the function to expand that form.
(struct transformer (name function) #:transparent)

; Convenience form to define a macro by pattern matching.

(module+ test
  (define-transformer m ++ [`(,_ ,<id>) `(set! ,<id> (+ ,<id> 1))])
  (check-equal? (transformer-name m) '++)
  (check-equal? ((transformer-function m) '(++ x)) '(set! x (+ x 1))))

(define-syntax-rule
  (define-transformer id name
    clause
    ...)
  (define id (transformer 'name (λ (e) (match e
                                         clause
                                         ...)))))

; Expand expression e with the list of transformers env.
(define (expand e env)

  ; Turn the list of transformers into a dictionary. Details not important.
  (define env′ (for/hash ([t env]) (values (transformer-name t) (transformer-function t))))

  (local [(define (expand e)
            
            (match e
              
              ; Head identifier determines meaning: pre-order traversal.

              ; New forms.
              [`(,head . ,_) #:when (dict-has-key? env′ head)
                             ; Internal node can re-arrange subtree, in particular its subtrees.
                             (expand ; Expansion continues after rewrite.
                              ; Rewrite:
                              ((dict-ref env′ head) e))]
              
              ; Core forms: they determine which sub-forms are expressions to continue expanding.
              [`(L0: datum ,_) e]
              [`(L0: var ,_) e]
              [`(L0: λ (,id) ,e) `(L0: λ (,id) ,(expand e))]
              [`(L0: app ,e0 ,e1) `(L0: app ,(expand e0) ,(expand e1))]
              [`(L0: set! ,id ,e) `(L0: set! ,id ,(expand e))]
              [`(L0: if ,e0 ,e1 ,e2) `(L0: if ,(expand e0) ,(expand e1) ,(expand e2))]

              ; Name default function application explicitly.
              [`(,head . ,_) (expand `(*app* . ,e))]

              ; Wrap identifier access and integer literals.
              [id #:when (symbol? id) (expand `(*id* ,id))]
              [integer (expand `(*datum* ,integer))]))]
    
    (expand e)))
