package w7lab;

import java.util.ArrayList;
import java.util.List;

public class Lab7 {

	/**
	 * Demonstrate usage of the various waiting lists.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		IWaitList<Integer> wl1 = new WaitList<>();
		IWaitList<Integer> wl2 = new BoundedWaitList<>(5);
		UnfairWaitList<Integer> wl3 = new UnfairWaitList<>();
		System.out.println(wl1); // []
		System.out.println(wl2); // []. Capacity 5
		System.out.println(wl3); // []

		for (int i = 0; i < 7; i++) {
			wl1.add(i);
			wl2.add(i);
			wl3.add(i);
		}
		System.out.println(wl1); // [0, 1, 2, 3, 4, 5, 6]
		System.out.println(wl2); // [0, 1, 2, 3, 4]. Capacity 5
		System.out.println(wl3); // [0, 1, 2, 3, 4, 5, 6]

		wl3.moveToBack(3);
		System.out.println(wl3); // [0, 1, 2, 4, 5, 6, 3]
		List<Integer> numbersEmptied = emptyWaitListOfCount(wl3, 4); // []
		System.out.println(numbersEmptied); // 0, 1, 2, 4
		System.out.println(wl3); // 5, 6, 3

		emptyWaitList(wl1);
		emptyWaitList(wl2);
		emptyWaitList(wl3);

		System.out.println(wl1); // []
		System.out.println(wl2); // []
		System.out.println(wl3); // []
	}

	/**
	 * Remove all items from wl. A generic method!
	 * 
	 * @param wl
	 *            the wait list to empty
	 */
	public static void emptyWaitList(IWaitList<?> wl) {
		while (!wl.isEmpty()) {
			wl.remove();
		}
	}

	/**
	 * Remove count items from waitList and return a list of the removed items.
	 * 
	 * A more complicated generic method. Notice that we put <T> before the
	 * return type to tell the function that we will be using that specific
	 * generic for the remainder of its definition.
	 *
	 * If we tried to use <?> as above, we wouldn't be able to define a return
	 * type like this. See
	 * https://docs.oracle.com/javase/tutorial/extra/generics/methods.html for
	 * more information.
	 * 
	 * @param waitList
	 *            the wait list to remove items from
	 * @param count
	 *            the number of items to remove
	 */
	public static <T> List<T> emptyWaitListOfCount(WaitList<T> waitList, int count) {
		List<T> ret = new ArrayList<>();
		for (int i = 0; i < count; i++) {
			if (waitList.isEmpty()) {
				break;
			}

			ret.add(waitList.remove());
		}
		return ret;
	}
}
