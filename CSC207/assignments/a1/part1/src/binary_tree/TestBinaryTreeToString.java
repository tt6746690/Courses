package binary_tree;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import java.lang.reflect.*;

/**
 * Test BinaryTree.toString.
 */
public class TestBinaryTreeToString {

	/** The tree to test. */
	private BinaryTree tree;

	/**
	 * BinaryTree.root. We'll use reflection to access it because it's private.
	 */
	private Field rootField;

	/**
	 * Create the binary tree to test and use reflection to make Field rootField
	 * non-private.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws SecurityException
	 * @throws NoSuchMethodException
	 * @throws NoSuchFieldException
	 */
	@Before
	public void setUp() throws NoSuchMethodException, SecurityException, NoSuchFieldException {
		tree = new BinaryTree();

		rootField = BinaryTree.class.getDeclaredField("root");
		rootField.setAccessible(true);
	}

	/**
	 * Test an empty tree.
	 */
	@Test
	public void testEmptyTree() {
		String expected = "()";
		String result = tree.toString();
		assertEquals(expected, result);
	}

	/**
	 * Test a tree with only a root node.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 */
	@Test
	public void testOneNode() throws IllegalArgumentException, IllegalAccessException {
		BinaryNode root = new BinaryNode("root");

		// Equivalent to tree.root = root;
		rootField.set(tree, root);

		String expected = "(root () ())";
		String result = tree.toString();
		assertEquals(expected, result);
	}

	/**
	 * Test a tree with a root and one left child.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 */
	@Test
	public void testOneLeftChild() throws IllegalArgumentException, IllegalAccessException {

		// Build the node structure.
		BinaryNode root = new BinaryNode("rootVal");
		BinaryNode left = new BinaryNode("leftVal");
		root.left = left;

		// Equivalent to tree.root = root;
		rootField.set(tree, root);

		String expected = "(rootVal (leftVal () ()) ())";
		String result = tree.toString();
		assertEquals(expected, result);

	}

	/**
	 * Test a tree with a root and one right child.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 */
	@Test
	public void testOneRightChild() throws IllegalArgumentException, IllegalAccessException {

		// Build the node structure.
		BinaryNode root = new BinaryNode("rootVal");
		BinaryNode right = new BinaryNode("rightVal");
		root.right = right;

		// Equivalent to tree.root = root;
		rootField.set(tree, root);

		String expected = "(rootVal () (rightVal () ()))";
		String result = tree.toString();
		assertEquals(expected, result);
	}

	/**
	 * Test a tree with a root and two children.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 */
	@Test
	public void testBothChildren() throws IllegalArgumentException, IllegalAccessException {

		// Build the node structure.
		BinaryNode root = new BinaryNode("rootVal");
		BinaryNode left = new BinaryNode("leftVal");
		BinaryNode right = new BinaryNode("rightVal");
		root.left = left;
		root.right = right;

		// Equivalent to tree.root = root;
		rootField.set(tree, root);

		String expected = "(rootVal (leftVal () ()) (rightVal () ()))";
		String result = tree.toString();
		assertEquals(expected, result);
	}

	/**
	 * Tree with several levels.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 */
	@Test
	public void testSeveralDeep() throws IllegalArgumentException, IllegalAccessException {

		// Build the node structure.
		BinaryNode root = new BinaryNode("rootVal");
		BinaryNode left1 = new BinaryNode("left1");
		BinaryNode left2 = new BinaryNode("left2");
		BinaryNode left3 = new BinaryNode("left3");
		BinaryNode right4 = new BinaryNode("right4");
		BinaryNode right = new BinaryNode("rightVal");
		root.left = left1;
		left1.left = left2;
		left2.left = left3;
		left3.right = right4;
		root.right = right;

		// Equivalent to tree.root = root;
		rootField.set(tree, root);

		String expected = "(rootVal (left1 (left2 (left3 () (right4 () ())) ()) ()) (rightVal () ()))";
		String result = tree.toString();
		assertEquals(expected, result);
	}

}
