
__Java Aliasing__

Having more than one copy of a pointer or reference to the same block of memory or object; When an object stored in memory having more than one variable referencing it. Since the variables store only references to the real object. Operations that mutates the object will be seen when trying to access any of the variables pointing to it.



```java
Rectangle box1 = new Rectangle (0, 0, 100, 200);
Rectangle box2 = box1;

System.out.println (box2.width); // 100
box1.grow (50, 50);              // width of box1 += 100
System.out.println (box2.width)  // 200
```

Note `box1` and `box2` are pointing to the same object
