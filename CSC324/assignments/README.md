


#### A1


```scheme
(format-symbol "make-~a" 'triple)
```


```scheme
'((λ3 (closure: (λ (f) (λ (g) (λ (h) (g ((f g) h))))) ((One λ1)))) 
  (λ2 (closure: (λ (Add1) (Add1 (Add1 One))) ((One λ1)))) 
  (λ1 (closure: (λ (h) h) ())) 
  (λ0 (closure: (λ (One) ((λ (Add1) (Add1 (Add1 One))) (λ (f) (λ (g) (λ (h) (g ((f g) h))))))) ())))
'Add1
```