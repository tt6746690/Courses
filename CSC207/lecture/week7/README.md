
Testing problem

1. visibility problem; private methods cannot be tested.  
2. coupling issues; if method depends on other methods, do not know which one is the culprit  
  + multiple entry point
  + create instances of interfaces to test s




JFrame

[doc](https://docs.oracle.com/javase/tutorial/uiswing/components/frame.html)


A frame, implemented as an instance of the JFrame class, is a window that has decorations such as a border, a title, and supports button components that close or iconify the window.


```java
//1. Create the frame. and set title
JFrame frame = new JFrame("FrameDemo");

//2. Optional: What happens when the frame closes?
frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

//3. Create components and put them in the frame.
//...create emptyLabel...
frame.getContentPane().add(emptyLabel, BorderLayout.CENTER);

//4. Size the frame. so that all contents are at or above their preferred sizes
frame.pack();

//5. Show it.
frame.setVisible(true);
```

Object equality
+ a good [resource](http://www.javaworld.com/article/2072762/java-app-dev/object-equality.html)



__Serializable__


_serialization_ is the process of translating data structures or object state into a format that can be stored and reconstructed lated in the same or another computer enviornment.

To __serialize__ an object means to convert its state to a byte stream so that the byte stream can be reverted back into a copy of the object. A Java object is serializable if its class or any of its superclasses implements either the java.io.Serializable interface or its subinterface, java.io.Externalizable. __Deserialization__ is the process of converting the serialized form of an object back into a copy of the object. Specifically a serialized object can be stored in a file and be later restored..

+ The serialization runtime associates with each serializable class a version number, called a `serialVersionUID`, which is used during deserialization to verify that the sender and receiver of a serialized object have loaded classes for that object that are compatible with respect to serialization. If the receiver has loaded a class for the object that has a different serialVersionUID than that of the corresponding sender's class, then deserialization will result in an  `InvalidClassException`


[tutorialpoint](https://www.tutorialspoint.com/java/java_serialization.htm)

Classes `ObjectInputStream` and `ObjectOutputStream` are high-level streams that contain the methods for serializing and deserializing an object.

```java
// serializes an Object and sends it to the output stream.
public final void writeObject(Object x) throws IOException

// retrieves next Object out of the stream and deserializes it.
public final Object readObject() throws IOException, ClassNotFoundException
```

A class serializes properly if
+ The class must implement the `java.io.Serializable` interface.  
+ All of the fields in the class must be serializable. If a field is not serializable, it must be marked `transient`.  



__Serialization example__  

```java
// a file named employee.ser is created.
import java.io.*;
public class SerializeDemo {

   public static void main(String [] args) {
      Employee e = new Employee();
      e.name = "Reyan Ali";
      e.address = "Phokka Kuan, Ambehta Peer";
      e.SSN = 11122333;
      e.number = 101;

      try {
         FileOutputStream fileOut =
         new FileOutputStream("/tmp/employee.ser");
         ObjectOutputStream out = new ObjectOutputStream(fileOut);
         out.writeObject(e);
         out.close();
         fileOut.close();
         System.out.printf("Serialized data is saved in /tmp/employee.ser");
      }catch(IOException i) {
         i.printStackTrace();
      }
   }
}
```

__Deserialization example__

```java
import java.io.*;
public class DeserializeDemo {

   public static void main(String [] args) {
      Employee e = null;
      try {
         FileInputStream fileIn = new FileInputStream("/tmp/employee.ser");
         ObjectInputStream in = new ObjectInputStream(fileIn);
         e = (Employee) in.readObject();
         in.close();
         fileIn.close();
      }catch(IOException i) {
         i.printStackTrace();
         return;
      }catch(ClassNotFoundException c) {      // have to catch this...
         System.out.println("Employee class not found");
         c.printStackTrace();
         return;
      }

      System.out.println("Deserialized Employee...");
      System.out.println("Name: " + e.name);
      System.out.println("Address: " + e.address);
      System.out.println("SSN: " + e.SSN);
      System.out.println("Number: " + e.number);
   }
}
```


Java has two kinds of classes for input and output (I/O):  
+ streams  
+ readers/writers.

Streams (InputStream, OutputStream and everything that extends these) are for reading and writing binary data from files, the network, or whatever other device.

Readers and writers are for reading and writing text (characters). They are a layer on top of streams, that converts binary data (bytes) to characters and back, using a character encoding.

Reading data from disk byte-by-byte is very inefficient. One way to speed it up is to use a buffer: instead of reading one byte at a time, you read a few thousand bytes at once, and put them in a buffer, in memory. Then you can look at the bytes in the buffer one by one.

```java
BufferedReader br=new BufferedReader(new InputStreamReader(System.in));
```


__`Java.io.BufferedInputStream`__  

The Java.io.BufferedInputStream class adds functionality to another input stream, the ability to buffer the input and to support the mark and reset methods.  
+ an internal buffer array is created
+ As bytes from the stream are read or skipped, the internal buffer is refilled as necessary from the contained input stream, many bytes at a time.


To convert unbuffered stream to buffered stream

```java
inputStream = new BufferedReader(new FileReader("xanadu.txt"));
outputStream = new BufferedWriter(new FileWriter("characteroutput.txt"));
```
