#lang racket

#| Require this library to print snapshots of a memory model for your code while it runs.

 The model is a tree of closures and environments, with the current environment double-boxed. |#


#| Traced operations.

 1. Closure creation.

   Fixed and variable arity (but not: mixed arity, default arguments, nor keyword arguments).

 2. Closure application.

   (Racket functions can be called as well, but their evaluation is not shown.)

 3. Definition.

   Modelled properly for a relatively simple but convenient extension of the Lambda Calculus.
    For the model to match the code's evaluation:

     Place definitions only at the top level of a program or closure.

     Do not shadow parameters from the immediately enclosing closure, since the parameter
      and body environments are merged to reduce the number of levels.

     Do not evaluate a reference to defined variable that has not completed initialization.

   The shorthand for implicit closure creation, with fixed and variable arity, is modelled.

 4. Variable mutation. |#


#| Control of printing. |#
(provide wait! wait-declaration! interleave! scale! body-width!)
#;{(wait! wait?)             ; boolean ‘wait?’ turns on/off pause for input between snapshots
   (wait-declaration! wait?) ; boolean ‘wait?’ turns on/off pause for input after declarations
   (interleave! interleave?) ; boolean ‘interleave?’ turns on/off closures interleaved with vs
   ;  pointing into the environment tree.
   (scale! font-size)        ; set font size
   (scale!)                  ; get font size
   (body-width! width)       ; set width for displayed closure bodies, in number of characters
   (body-width!)}            ; get width for displayed closure bodies


#| Syntax of the traced expression and definition forms. |#
(provide (rename-out (traced:λ λ) (traced:λ lambda)
                     (traced:define define)
                     (traced:set! set!)))
#;{; Closure creation.
   (λ (<formal-identifier> ...) ; or ‘lambda’
     <body-expression/definition>
     ...
     <body-expression>)
   (λ <formal-identifier>       ; or ‘lambda’
     <body-expression/definition>
     ...
     <body-expression>)
   ; Also: implicitly from function definition form of ‘define’.
   
   ; Closure application.
   (<closure-expression> <argument-expression> ...)
   ; Also: implicitly when passed to higher-order library functions such as ‘map’ and ‘apply’.
   
   ; Definition.
   (define <variable-identifier> <initialization-expression>)
   (define (<variable-identifier> <formal-identifier> ...)
     <body-expression/definition>
     ...
     <body-expression>)
   
   ; Variable mutation.
   (set! <variable-identifier> <expression>)}


#| Implementation of the overriden forms. |#

(define-syntax traced:define
  (syntax-rules ()
    [(traced:define (#:name <variable-identifier> <formal-identifier> ...)
                    <body-expression/definition> ... <body-expression>)
     (traced:define <variable-identifier>
                    (traced:λ #:name <variable-identifier> (<formal-identifier> ...)
                              <body-expression/definition> ... <body-expression>))]
    [(traced:define (<variable-identifier> <formal-identifier> ...)
                    <body-expression/definition> ... <body-expression>)
     (traced:define <variable-identifier>
                    (traced:λ (<formal-identifier> ...)
                              <body-expression/definition> ... <body-expression>))]
    [(traced:define <variable-identifier> <initialization-expression>)
     (begin (declare-id! '<variable-identifier>)
            (define <variable-identifier> undefined) (when (wait-declaration!) (show))
            ;
            (traced:set! <variable-identifier> <initialization-expression>))]))

(define-syntax-rule (traced:set! <variable-identifier> <expression>)
  (set! <variable-identifier> (let ([expression-value <expression>])
                                (set-id! '<variable-identifier> expression-value) (show)
                                expression-value)))

(define-syntax Lambda
  (syntax-rules ()
    [(Lambda <name> <formals> <body> <arguments>)
     (letrec ([λ<n> (closure-create! '<name> '<formals> '<body>
                                     (λ <formals>
                                       (call! λ<n> <arguments>) (show)
                                       (define expression-value (return! (λ () . <body>))) (show)
                                       expression-value))])
       (show)
       λ<n>)]))

(define-syntax traced:λ
  (syntax-rules ()
    [(traced:λ #:name <name>
               (<formal-identifier> ...) <body-expression/definition> ... <body-expression>)
     (Lambda <name> (<formal-identifier> ...) (<body-expression/definition> ... <body-expression>)
             (list <formal-identifier> ...))]
    [(traced:λ #:name <name>
               <formal-identifier> <body-expression/definition> ... <body-expression>)
     (Lambda <name> <formal-identifier> (<body-expression/definition> ... <body-expression>)
             <formal-identifier>)]
    [(traced:λ (<formal-identifier> ...) <body-expression/definition> ... <body-expression>)
     (Lambda #false (<formal-identifier> ...) (<body-expression/definition> ... <body-expression>)
             (list <formal-identifier> ...))]
    [(traced:λ <formal-identifier> <body-expression/definition> ... <body-expression>)
     (Lambda #false <formal-identifier> (<body-expression/definition> ... <body-expression>)
             <formal-identifier>)]))

(module model racket
  
  #| Tree's Datatype. |#
  (struct Tree (uid parent [children #:mutable]) #:transparent)
  (struct Closure Tree (hide? formals body) #:transparent)
  (struct Environment Tree (bindings) #:transparent)
  ;
  (define (extend-children! a-tree child)
    (set-Tree-children! a-tree (append (Tree-children a-tree) (list child))))
  
  #| State: global and current environment, and whole stack. |#
  (define environment-create!
    (let ([uid 0])
      (λ (parent) (local-require (only-in racket/syntax format-symbol))
        (begin0 (Environment (if (symbol? parent)
                                 '||
                                 (format-symbol "Called ~a with:" (Tree-uid parent)))
                             parent (list) (make-hash))
                (set! uid (add1 uid))))))
  ;
  (define current-environment (environment-create! 'GLOBAL))
  (define global-environment current-environment)
  (define call-stack (list))
  ;
  (define-values (push-current-environment! pop-current-environment!)
    (let ([call-stack (list)])
      (values (λ () (set! call-stack (cons current-environment call-stack)))
              (λ () (set!-values (current-environment call-stack)
                                 (values (first call-stack) (rest call-stack)))))))
  
  #| Closures: user's versus racket's. |#
  (define our-closures (make-hasheq))
  (define (convert-if-closure c) (hash-ref our-closures c (λ () c)))
  
  #| Core forms. |#
  
  ; Closure creation.
  (provide closure-create!)
  (define closure-create!
    (let ([uid 0])
      (λ (name formals body function)
        (local-require (only-in racket/syntax format-symbol))
        (let ([model-closure (Closure (format-symbol "λ~a.~a" uid (if name name ""))
                                      current-environment (list) name formals body)])
          (set! uid (add1 uid))
          (define λ<n> (procedure-rename function (Tree-uid model-closure)))
          (hash-set! our-closures λ<n> model-closure)
          (extend-children! current-environment λ<n>)
          λ<n>))))
  
  ; Closure application.
  (provide call! return!)
  (define (call! closure arguments)
    (push-current-environment!)
    (define model-closure (convert-if-closure closure))
    (define frame (environment-create! model-closure))
    (extend-children! model-closure frame)
    (set! current-environment frame)
    (define formals (Closure-formals model-closure))
    (cond [(symbol? formals) (declare-id! formals)
                             (set-id! formals arguments)]
          [else (for ([parameter (Closure-formals model-closure)]
                      [argument arguments])
                  (declare-id! parameter)
                  (set-id! parameter argument))]))
  (define (return! body-thunk)
    (begin0 (body-thunk)
            (pop-current-environment!)))
  
  ; Definition.
  (provide undefined declare-id!)
  (define undefined 'undefined)
  (define (declare-id! variable-name)
    (hash-set! (Environment-bindings current-environment) variable-name undefined))
  
  ; Variable mutation.
  (provide set-id!)
  (define (set-id! variable-name expression-value [environment current-environment])
    (let loop ([environment environment])
      (if (and (Environment? environment)
               (hash-has-key? (Environment-bindings environment) variable-name))
          (hash-set! (Environment-bindings environment) variable-name expression-value)
          (loop (Tree-parent environment)))))
  
  
  #| Control. |#
  
  (provide wait! wait-declaration! interleave! scale! body-width!)
  (provide show)
  
  (define wait! (make-parameter #true))
  (define wait-declaration! (make-parameter #false))
  
  (define (show)
    (when (wait!) (read-line))
    (print (semantic-model global-environment)) (newline))
  
  (require (only-in slideshow/code get-current-code-font-size))
  (define (scale! [size #false])
    (if size
        (get-current-code-font-size (λ () size))
        ((get-current-code-font-size))))
  (scale! 12)

  (define interleave! (make-parameter #true))
  
  #| Drawing. |#
  
  (provide semantic-model)
  
  (module Drawing slideshow
    (define body-width! (make-parameter 40))
    (define inset′ (curryr inset 3 0))
    (provide (all-defined-out)
             vl-append hbl-append
             blank rounded-rectangle rectangle
             code)
    (require (only-in slideshow/code code current-code-tt get-current-code-font-size))
    (define (scale) ((get-current-code-font-size)))
    (define (code-label code)
      (inset′ (apply vl-append
                     (map (current-code-tt)
                          (regexp-split "\n"
                                        (call-with-output-string
                                         (λ (port)
                                           (parameterize ([pretty-print-columns (body-width!)])
                                             (pretty-print code port 1)))))))))
    (define (label token) (inset′ (code (unsyntax token))))
    (define (boxify picture [shape rectangle])
      (cc-superimpose picture (shape (+ (scale) (pict-width picture))
                                     (+ (scale) (pict-height picture)))))
    (define (layout-siblings siblings) (apply ht-append (scale) siblings))
    (define (layout-parent-children parent children) (vc-append (* 3 (scale)) parent children))
    (define (join! tree parent child)
      (pin-arrow-line (scale) tree
                      child ct-find
                      parent cb-find
                      #:start-pull 1/10 #:start-angle (/ pi 2)
                      #:end-pull 1/20 #:end-angle (/ pi 2))))
  (require 'Drawing)
  
  (define (closure-name c) (label (Tree-uid c)))
  (define (semantic-model e/c)
    (let* ([e/c (if (Environment? e/c) e/c (dict-ref our-closures e/c))]
           [root (boxify
                  (apply vl-append
                         (if (Closure? e/c)
                             (cons (hbl-append (closure-name e/c)
                                               (label (Closure-formals e/c)))
                                   (if (Closure-hide? e/c) '() (map code-label (Closure-body e/c))))
                             (cons (if (interleave!) (blank) (label (Tree-uid e/c)))
                                   (for/list ([(id value) (Environment-bindings e/c)])
                                     (hbl-append (label id) (label '|: |)
                                                 ((if (Closure? value) closure-name label)
                                                  value))))))
                  (if (Closure? e/c) rounded-rectangle rectangle))]
           [children (map semantic-model
                          (if (interleave!)
                              (Tree-children e/c)
                              (if (Environment? e/c)
                                  (append (Tree-children e/c)
                                          (append-map Tree-children (map (curry dict-ref our-closures)
                                                                         (Tree-children e/c))))
                                  '())))]
           [arranged (layout-parent-children
                      (if (eq? e/c current-environment) (boxify root) root)
                      (layout-siblings children))])
      (for/fold ([tree arranged])
                ([child children])
        (join! tree root child)))))

(require 'model)