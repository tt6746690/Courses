package w7lab;

import java.util.Collection;
import java.util.Collections;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

// WaitList is a generic class. 
//
// We say that WaitList is a parameterized type and E is the type variable 
// (or type parameter). 
//
// Typically with generics, we use T (stands for type) for the type variable. 
// But for collections, we use E (stands for element), as in the Java API.

/**
 * A representation of a waiting list. Works in a first-come first-serve basis.
 * 
 * @author campbell
 * @author t6charti
 */

public class WaitList<E> implements IWaitList<E>{

	/** The waitlist contents. */
	protected Queue<E> content;
	// check out documentation at:
	// http://docs.oracle.com/javase/8/docs/api/java/util/concurrent/ConcurrentLinkedQueue.html

	/**
	 * Create a new empty WaitList.
	 */
	public WaitList() {
		// we re-use the content creation code in our other constructor by
		// giving it an empty collection as the initial contents of this
		// WaitList.
		this(Collections.emptyList());
	}

	/**
	 * Create a new WaitList containing all the elements from c.
	 * 
	 * @param c
	 *            elements from this Collection are added to the new WaitList
	 */
	public WaitList(Collection<E> c) {
		this.content = new ConcurrentLinkedQueue<E>(c); // queue having element of type <E>
	}

	/**
	 * Add element to this WaitList.
	 * 
	 * @param element
	 *            the new element to be added
	 */
	@Override
	// you must Override a method in a parent interface or class for this
	// warning to go away.
	public void add(E element) {
		this.content.add(element);
	}

	/**
	 * Remove and return the next element from this WaitList. The elements are
	 * removed in first-in-first-out order. Return null if this WaitList is
	 * empty.
	 * 
	 * @return the next element in this WaitList, or null if this WaitList is
	 *         empty
	 */
	@Override
	public E remove() { // similar: return type is E
		return this.content.poll();
	}

	/**
	 * Return true if and only if this WaitList contains element.
	 * 
	 * @param element
	 *            the element to test for membership
	 * @return true if this WaitList contains element and false otherwise
	 */
	@Override
	public boolean contains(E element) {
		return this.content.contains(element);
	}

	/**
	 * Return whether this wait list contains all the items in c.
	 * 
	 * @param c
	 *            the Collection of elements to test for membership in this
	 *            WaitList
	 * @return true if this WaitList contains all elements in the given
	 *         collection and false otherwise
	 */
	@Override
	public boolean containsAll(Collection<E> c) {
		return this.content.containsAll(c);
	}

	/**
	 * Return whether this wait list is empty.
	 * 
	 * @return true if and only if this WaitList has no elements
	 */
	@Override
	public boolean isEmpty() {
		return this.content.isEmpty();
	}

	@Override
	/**
	 * Return a string representation of this wait list.
	 * 
	 * @return a string representation of this wait list
	 */
	public String toString() {
		return this.content.toString();
	}
}
