package iterator;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * An address book: a list of contacts.
 */
class AddressBook implements Iterable<Contact> {

    /** This AddressBook's list of contacts. */
    private List<Contact> contacts;

    /**
     * Constructs a new AddressBook.
     */
    public AddressBook() {
        contacts = new ArrayList<>();
    }

    /**
     * Adds a new contact with last name lastName, first name firstName, email
     * address email, phone number phone, and address address.
     * @param lastName the last name
     * @param firstName the first name
     * @param email the email address
     * @param phone the phone number
     * @param address the address
     */
    public void addContact(String lastName, String firstName,
            String email, String phone, String address) {
        Contact c = new Contact(lastName, firstName, email, phone, address);
        contacts.add(c);
    }

    /**
     * Returns the number of contacts in this book.
     * @return the number of contacts in this book.
     */
    public int getSize() {
        return contacts.size();
    }

    @Override
    public String toString() {
        String res = "";
        for (Contact c : contacts) {
            res += c;
        }
        return res;
    }

    /**
     * Returns an iterator for this address book.
     * @return an iterator for this address book.
     */
    @Override
    public Iterator<Contact> iterator() {
        return new AddressBookIterator();
    }

    /**
     * An Iterator for AddressBook Contacts.
     */
    private class AddressBookIterator implements Iterator<Contact> {

        /** The index of the next Contact to return. */
        private int current = 0;

        /**
         * Returns whether there is another Contact to return.
         * @return whether there is another Contact to return.
         */
        @Override
        public boolean hasNext() {
            return current < contacts.size();
        }

        /**
         * Returns the next Contact.
         * @return the next Contact.
         */
        @Override
        public Contact next() {
            Contact res;

            // List.get(i) throws an IndexOutBoundsException if
            // we call it with i >= contacts.size().
            // But Iterator's next() needs to throw a 
            // NoSuchElementException if there are no more elements.
            try {
                res = contacts.get(current);
            } catch (IndexOutOfBoundsException e) {
                throw new NoSuchElementException();
            }
            current += 1;
            return res;
        }

        /**
         * Removes the contact just returned.  Unsupported.
         */
        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not supported.");
        }
    }
}
