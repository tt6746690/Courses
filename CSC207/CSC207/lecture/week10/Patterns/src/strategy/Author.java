package strategy;

import java.util.ArrayList;
import java.util.List;

public class Author {
    
    private String name;         // this Author's name
    private List<Book> books;    // this Author's books
    private Sorter<Book> sorter; // this Author's sorting strategy

    
    /**
     * Constructs a new Author named name that uses sorting strategy sorter
     *  to sort books.
     * @param name the name of the new Author
     * @param sorter the sorting strategy used to sort books
     */
    public Author(String name, Sorter<Book> sorter) {
        this.setName(name);
        this.books = new ArrayList<Book>();
        this.sorter = sorter;
    }
    
    /**
     * Returns this Author's name.
     * @return this Author's name
     */
    public String getName() {
    return name;
    }
    
    /**
     * Sets this Author's name to name.
     * @param name this Author's new name
     */
    public void setName(String name) {
    this.name = name;
    }
    
    /**
     * Adds book to this Author's list of books.
     * @param book a book to be added to this Author's books
     */
    public void addBook(Book book) {
    books.add(book);
    }
    
    /**
     * Sorts this Author's books.
     */
    public void sortBooks() {
    sorter.sort(books);
    }
    
    @Override
    public String toString() {
    return name + ": " + books.toString();
    }
}
