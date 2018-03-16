#lang racket #| Garbage Collection |#


#; ((λ2 (f) (f 2107))
    ((λ1 (c) (λ0 (_) c))
     488))

#| heap:  C0:[&λ2 •] C1:[&λ1 •] E0:[• 488] C2:[&λ0 &E0] E1:[• &C2] E2:[&E0 2107]

   env: •
   stack: &C0
          &C1
   result: 488

   ; About to do body of C1
   env: E0
   stack: &C0
          •
   result: -

   ; About to call C2
   env: •
   stack: &C0
   result: &C2

 heap:  C0:[&λ2 •] C1:[&λ1 •] E0:[• 488] C2:[&λ0 &E0]
     we can remove C1 since we no longer refer to it


   ; Modify to implement garbage collection
   env: •
   stack: &C0′
   result: &C2′

 heap:  C0:[&C0′ .11 -] C1:[&λ1 •] E0:[&E0′ .11 -] C2:[&C2′ .11 -] E3:[&E0 &C0]


 "root set of pointers" : pointers in the registers and stack
 C0′:[&λ2 •] C2′:[&λ0 &E0′] E0′:[• 488] E3:[&E0′ &C0′]

 For 64 bits: word size is 8 bytes, if we stay aligned then all pointers
 into the heap are multiples of 8 : binary _____000

 Dynamic Typing [runtime type information] (value has information of the type)
 _____________000 : pointer into the heap
 _____________001 : integer, actual value : divide by 8
 0000000000000010 : boolean false
 1111111111111010 : boolean false
 _____________011 : "forwarding address" during garbage collection

 That was "two-space" copying garbage collection
   Needs twice as much memory for the heap
   Essentially no cost for allocation
   No fragmentation
   Handles cycles
   Time O("live" data), not of garbage or whole heap
   not "incremental", so fewer larger pauses (amortized)
 


 •·⋯·•·⋯·•·⋯·•·⋯·•·⋯·•·⋯·•·⋯·•·⋯·•·⋯·•·⋯·•
 ••⋯·⋯·····⋯·⋯••⋯·⋯·····⋯·⋯••⋯·⋯·····⋯·⋯••⋯·⋯·····⋯·⋯•
 ⋯·⋯⋯·⋯⋯⋯·⋯⋯⋯⋯·⋯⋯⋯⋯⋯·⋯⋯⋯⋯⋯⋯·⋯⋯⋯⋯⋯⋯⋯·⋯⋯⋯⋯⋯⋯·⋯⋯⋯⋯⋯·⋯⋯⋯⋯·⋯⋯⋯·⋯⋯·⋯
 •••⋯•••⋯•••⋯•••⋯•••⋯•••⋯•••⋯•••⋯•••⋯•••⋯•••⋯•••⋯•••⋯•••⋯•••⋯•••⋯•••
 •·••·•••·••••·•••••·••••••·•••••••·••••••••·•••••••••·••••••••••·•••••••••••


 ←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→←→
 ΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩωΩ
 ΞΓΞLΞΓΞLΞΓΞLΞΓΞLΞΓΞLΞΓΞLΞΓΞLΞΓΞLΞΓΞLΞΓΞLΞΓΞLΞΓΞLΞΓΞL
 ξζζξξζζξξζζξξζζξξζζξξζζξξζζξξζζξξζζξξζζξξζζξξζζξξζζξ


|#