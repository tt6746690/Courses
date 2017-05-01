__Design Pattern__



__Java Set__  
A collection that contains no duplicate elements. More formally, sets contain no pair of elements e1 and e2 such that `e1.equals(e2)`, and at most one null element. As implied by its name, this interface models the mathematical set abstraction.


__HashSet__
This class implements the `Set` interface, backed by a hash table (actually a HashMap instance). It makes no guarantees as to the iteration order of the set; in particular, it does not guarantee that the order will remain constant over time. This class permits the null element.

1. HashSet doesn’t maintain any order, the elements would be returned in any random order.
2. HashSet doesn’t allow duplicates. If you try to add a duplicate element in HashSet, the old value would be overwritten.
3. HashSet allows null values however if you insert more than one nulls it would still return only one null value.
4. HashSet is non-synchronized.
5. The iterator returned by this class is fail-fast which means iterator would throw ConcurrentModificationException if HashSet has been modified after creation of iterator, by any means except iterator’s own remove method.


```java
import java.util.HashSet;
public class HashSetExample {
   public static void main(String args[]) {
      // HashSet declaration
      HashSet<String> hset =
               new HashSet<String>();

      // Adding elements to the HashSet
      hset.add("Apple");
      hset.add("Mango");
      hset.add("Grapes");
      hset.add("Orange");
      hset.add("Fig");
      //Addition of duplicate elements
      hset.add("Apple");
      hset.add("Mango");
      //Addition of null values
      hset.add(null);
      hset.add(null);

      //Displaying HashSet elements
      System.out.println(hset);
    }
}

// outputs: [null, Mango, Grapes, Apple, Orange, Fig]
```


__ArrayList__

```java
ArrayList<String> obj = new ArrayList<String>();

obj.add("Harry");
obj.add(1, "Justin");
obj.remove("Harry");
obj.remove(1);
```
__Initialization__

```java
ArrayList<Type> list = new ArrayList<Type>(
        Arrays.asList(Object o1, Object o2, Object o3));
```

```java
ArrayList<T> list = new ArrayList<T>();
	   list.add("Object o1");
	   list.add("Object o2");
	   list.add("Object o3");
```


__sublist__

`List subList(int fromIndex, int toIndex)`
+ Here fromIndex is inclusive and toIndex is exclusive.

```java
ArrayList<String> sublist = new ArrayList<String>(obj.subList(1, 2));
```

The subList method throws `IndexOutOfBoundsException` – if the specified indexes are out of the range of ArrayList (`fromIndex < 0 || toIndex > size`).


__Join List__

```java
ArrayList<String> list1 = new ArrayList<String>();
ArrayList<String> list2 = new ArrayList<String>();
ArrayList<String> final = new ArrayList<String>();
final.addAll(list1);
final.addAll(list2);


```


__Linked List__

LinkedList is an implementation of List interface.

```java
LinkedList<String> llistobj  = new LinkedList<String>();
llistobj.add("Hello");
llistobj.add(2, "bye");
llistobj.addFirst("text");
llistobj.addLast("Chaitanya");

Object o = llistobj.poll(); // removes and returns first item
llistobj.remove();          // removes last item
llistobj.set(2, "Test");    // updates item at specified index

```



__Swing__

__ActionListener [link](http://docs.oracle.com/javase/tutorial/uiswing/events/actionlistener.html)__

To write an actionListener:

1. Declare an event handler class and specify that the class either implements an ActionListener interface or extends a class that implements an ActionListener interface.

```java
public class MyClass implements ActionListener { }
```

2. Register an instance of the event handler class as a listener on one or more components.

```java
someComponent.addActionListener(instanceOfMyClass);
```


3. Include code that implements the methods in listener interface.

```java
public void actionPerformed(ActionEvent e) {
    ...//code that reacts to the action...
}
```


For example,

```java
import java.awt.*;
import java.awt.event.*;

public class AL extends Frame implements WindowListener,ActionListener {
        TextField text = new TextField(20);
        Button b;
        private int numClicks = 0;

        public static void main(String[] args) {
                AL myWindow = new AL("My first window");
                myWindow.setSize(350,100);
                myWindow.setVisible(true);
        }

        public AL(String title) {

                super(title);
                setLayout(new FlowLayout());
                addWindowListener(this);
                b = new Button("Click me");
                add(b);
                add(text);
                b.addActionListener(this);
        }

        public void actionPerformed(ActionEvent e) {
                numClicks++;
                text.setText("Button Clicked " + numClicks + " times");
        }

        public void windowClosing(WindowEvent e) {
                dispose();
                System.exit(0);
        }

        public void windowOpened(WindowEvent e) {}
        public void windowActivated(WindowEvent e) {}
        public void windowIconified(WindowEvent e) {}
        public void windowDeiconified(WindowEvent e) {}
        public void windowDeactivated(WindowEvent e) {}
        public void windowClosed(WindowEvent e) {}

}

```



__Top level container class [link](http://docs.oracle.com/javase/tutorial/uiswing/components/toplevel.html)__

__JFrame__

A Frame is a top-level window with a title and a border. It also includes event handlers for various Events like `windowClose`, `windowOpened` etc.


__JPanel__

JPanel is a generic lightweight container. A generic container to group other components together
+ It is useful when working with LayoutManagers e.g. `GridLayout` f.i adding components to different `JPanels` which will then be added to the JFrame to create the gui. It will be more manageable in terms of Layout and re-usability.


```java
JFrame topLevelContainer = new JFrame("FrameDemo");

//Create a panel and add components to it.
JPanel contentPane = new JPanel(new BorderLayout());
contentPane.add(someComponent, BorderLayout.CENTER);
contentPane.add(anotherComponent, BorderLayout.PAGE_END);

topLevelContainer.setContentPane(contentPane);
topLevelContainer.pack();
topLevelContainer.setVisible(true);
```
__JComponent__

__JLabel__
A display area for a short text string or an image, or both.

```java
ImageIcon icon = createImageIcon("images/middle.gif");
. . .
label1 = new JLabel("Image and Text",
                    icon,
                    JLabel.CENTER);
//Set the position of the text, relative to the icon:
label1.setVerticalTextPosition(JLabel.BOTTOM);
label1.setHorizontalTextPosition(JLabel.CENTER);

label2 = new JLabel("Text-Only Label");
label3 = new JLabel(icon);
```
