Linked List
=========


### Regular list VS linked list

| | regular python list | linked list     |
| :-- |:------------- | :------------- |
| pro | it can efficiently be accessed      | reserves just enough memory for the object value they want to refer to, a reference to it, and a reference to the next node in the list. Therefore can efficiently grow and shrink as needed |   
|con| they allocate large blocks of _contiguous_ memory, which becomes increasingly difficult as memory is in use. | accessing an item is inefficient, as needs to traverse through the list via all the nodes before it |
|examples| class enrolment - since not many people added or dropped; printing the entire list is efficient | passenger baggages |


### Node
a node consists a `value` and `Next_`

### LinkedList   
linkedlist is a wrapper for the nodes. Linkedlist stores information of the __front__ (the first node), __back__ (the last list), and the __size__ of the linked list.

+ add a node to the front of the list
+ add a node to the back of the list (back is therefore very useful - don't have to traverse the entire list)
+ remove a node from the list
+ check if the list contains a specific value          
