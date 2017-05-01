// Week 4, lecture 2: Our first exposure to defining an interface.

package university;

// Our first version of the interface was specific.  It demanded that,
// in order to be considered "IDed", a class must identify itself with
// a String.  I.e., it must provide a method getID that returns a String.

//public interface IDed {
//    
//    public String getID();
//    
//}

// class Student implements IDed {
//     public String getID() {
//         return this.id;
//     }
// }

// Now we demand that, in order to be
// considered IDed by a value of type T, a class must provide a method
// getID that returns a T.
// Notice that type T is essentially a parameter to the interface.
// Cool!

public interface IDed<T> {
    
    public T getID();

}
