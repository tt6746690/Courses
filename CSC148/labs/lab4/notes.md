

### tricks

+ If were to delete a node from linkedlist, just change the reference, or `_next`, to skip the deleted node   
+ for two node to be equivalent, the reference need not be identical as long as the values are. For linkedlist, so long as `front`, `back`, and `size` are equivalent, linkedlists are identical
+ for deletions to the linkedlist, need to keep track of the size of the linkedlist `size -= 1`  
+ must create new nodes when appending or prepending to the linkedlist. just need to change the reference to the newly created nodes and adjust `front`, `back` correspondingly. Order matters!!

```python
self.back.next_ = new_node
self.back = new_node
```

+ Account for exceptions when the linkedlist is empty, when the size of linkedlist is 0.

+ To set item based on index. Apply the following to allow for negative index.

```python
while index < 0:
  index += self.size
```

+ account for index errors

```python
if index >= self.size:
  raise IndexError('index out of range')
```

+ Use for loops to access items by index

+ logic for adding two linkedlist together

```python
list1 = self.copy()
list2 = other.copy()
list1.back.next_ = list2.front
list1.back = list2.back
return list1
```

+ To get to the node based on value comparison. A neat way to separate logics.

```python
while current_node is not None and current_node.value != value2:
  previous_node = current_node
  current_node = current_node.next_
```
