#lang racket

(require "mm.rkt")
#;(scale! 20)
(body-width! 150)
#;(wait! #false)

#| public class AddressBook implements Iterable<Contact> {
     private List<Contact> contacts ;
     public AddressBook() { contacts = new ArrayList() ;}
     public void addContact(String name, String email, String phone) {
        contacts.add(new Contact(name, email, phone)) ;}
     public Iterator<Contact> iterator() { return new AddressBookIterator() ;}
     private class AddressBookIterator implements Iterator<Contact> {
        private int nextIndex = 0 ;
        public boolean hasNext() { return nextIndex != contacts.size() ;}
        public Contact next() {
            Contact next = contacts.get(nextIndex) ;
            nextIndex++ ;
            return next ;}}} |#


(define-syntax class
  (syntax-rules ()
    [(class (class-id init-id ...) ; Constructor.
       ; Initialization block to evaluate each time the constructor is called.
       [init
        ...]
       ; Groups of method header and body.
       [(method-id parameter-id ...) body
                                     ...]
       ...)
     (define (#:name class-id init-id ...)
       init
       ...
       (λ #:name class-id
         message+arguments
         (match message+arguments
           [`(method-id ,parameter-id ...) body
                                           ...]
           ...)))]))

(class (AddressBook) ; No-argument constructor.
  ; Object initialization includes making a variable and a class.
  [(define contacts '())
   ; An inner class is really a class generator, generating a class for each object.
   ; Instances of the generated class can then depend on the particular object that created
   ;  the generated class.
   (class (AddressBookIterator) ; No-argument constructor.
     ; Object initialization makes a variable.
     [(define nextIndex 0)]
     ; The two methods of AddressBookIterator.
     [(hasNext) (not (= nextIndex (length contacts)))]
     [(next) (define next (list-ref contacts nextIndex))
             (set! nextIndex (add1 nextIndex))
             next])]
  ; The two methods of AddressBook.
  [(addContact name email phone) (set! contacts (list* (list name email phone) contacts))]
  [(iterator) (AddressBookIterator)])

; (λ0.AddressBook () _) is the class.

(define ab1 (AddressBook))
; (λ2.AddressBook message+arguments _) is the new AddressBook object.
; (λ1.AddressBookIterator () _) is the corresponding inner class, in the scope of the
;  new AddressBook instance's variables.



(define ab2 (AddressBook))
; (λ4.AddressBook message+arguments _) is the new AddressBook object.
; (λ3.AddressBookIterator () _) is the corresponding inner class, in the scope of the
;  new AddressBook instance's variables.


(ab1 'addContact 'pgries 'paul '555-123-4567)

(define it1-1 (ab1 'iterator))
; (λ5.AddressBookIterator message+arguments _) is an instance of the inner class for λ2,
;  made by that inner class λ1, so with its own environment for ‘nextIndex’, and extending
;  λ1's environment which includes λ2's instance variable ‘contacts’.

(define it1-2 (ab1 'iterator))
; Similar, but with its own ‘nextIndex’.

(define it2 (ab2 'iterator))
; λ7 has its own ‘nextIndex’, but unlike the other two iterators, its environment extends
;  λ3's environment, which includes λ4's instance variable ‘contacts’.

(it1-1 'next)
; Changes the value of ‘nextIndex’ in λ5's local environment, and accesses ‘contacts’ in
;  its more global environment.
