package iterator;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Iterator;
import java.util.Scanner;

/**
 * An address book application.
 */
public class Main {

    /**
     * The name of the file in which the contacts are stored.
     */
    private static final String BOOK_FILE ="addresses.txt";

    /**
     * @param args the command line arguments
     * @throws FileNotFoundException if the address book file does not exist.
     */
    public static void main(final String[] args) 
            throws FileNotFoundException {
        AddressBook book = new AddressBook();
        readBook(book);

        // We can explicitly construct an iterator and go through it.
        Iterator<Contact> contacts0 = book.iterator();
        while (contacts0.hasNext()) {
            System.out.println(contacts0.next());
        }

        // Or we can use this style of for-loop (which depends upon an 
        // iterator).
        System.out.println("==========");
        for (Contact c : book) {
            System.out.println(c);
        }
        
        // In fact, for ANY Iterable<T> iterable, you can write a for-each 
        // loop in Java:
        // for (T item: iterable) {
        //    ... doSomething(item) ... 
        // }
        // the meaning of this for loop is:
        // Iterator<T> it = iterable.iterator();
        // T item;
        // while (it.hasNext()) {
        //   item = it.next();
        //   ... doSomething(item) ... 
        // }

        // Notice that we can have multiple iterators for the same address book,
        // and they are independent.  Advancing one does not advance the other.
        Iterator<Contact> contacts1 = book.iterator();
        Iterator<Contact> contacts2 = book.iterator();

        System.out.println("==========");
        System.out.println(String.format("C1: %s, %s", contacts1.hasNext(),
                contacts1.next()));
        System.out.println(String.format("C1: %s, %s", contacts1.hasNext(),
                contacts1.next()));
        System.out.println(String.format("C1: %s, %s", contacts1.hasNext(),
                contacts1.next()));
        System.out.println(String.format("C2: %s, %s", contacts2.hasNext(),
                contacts2.next()));

        System.out.println(String.format("C1: %s, %s", contacts1.hasNext(),
                contacts1.next()));
        System.out.println(String.format("C2: %s, %s", contacts2.hasNext(),
                contacts2.next()));
    }

    /**
     * Reads the address book file and add contacts to book.
     *
     * @param book the AddressBook to fill with Contacts.
     * @throws FileNotFoundException if the file does not exist.
     */
    private static void readBook(AddressBook book)
        throws FileNotFoundException {
      Scanner sc = new Scanner(new File(BOOK_FILE));
      while (sc.hasNextLine()) {
        String[] record = sc.nextLine().trim().split(",");
        book.addContact(record[0], record[1], record[2],
            record[3], record[4]);
      }
      sc.close();
    }
}