package strategy;

import java.io.FileNotFoundException;

public class Main {
    
    public static void main(String[] args) throws FileNotFoundException {
        
        Sorter<Book> sorter1 = new InsertionSorter<Book>();
        Sorter<Book> sorter2 = new SelectionSorter<Book>();

        Author author1 = new Author("Lynn Coady", sorter1);
        Author author2 = new Author("Dennis Bock", sorter2);
                
        Book b1 = new Book("Hellgoing: Stories", "1770893083");
        Book b2 = new Book("Going Home Again", "1443433659");
        Book b3 = new Book("The Antagonist", "1770891048");
        Book b4 = new Book("Mean Boy", "0385659768");
        Book b5 = new Book("The Ash Garden", "0006485456");
        
        author1.addBook(b1);
        author1.addBook(b3);
        author1.addBook(b4);
        
        author2.addBook(b2);
        author2.addBook(b5);
        
        author1.sortBooks();
        System.out.println(author1.toString());
        
        author2.sortBooks();
        System.out.println(author2.toString());

    }
}