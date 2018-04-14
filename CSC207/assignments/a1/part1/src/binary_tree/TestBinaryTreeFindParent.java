package binary_tree;

import static org.junit.Assert.*;

import org.junit.Before;
import org.junit.Test;
import java.lang.reflect.*;

public class TestBinaryTreeFindParent {

	/** The tree to test. */
	private BinaryTree tree;

	/**
	 * BinaryTree.findParent. We'll use reflection to access it because it's
	 * private.
	 */
	private Method findParent;

	/**
	 * BinaryTree.root. We'll use reflection to access it because it's private.
	 */
	private Field rootField;

	/**
	 * Create the binary tree to test and use reflection to make Method
	 * findParent and Field rootField non-private.
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

		// Make the private methods accessible.
		findParent = BinaryTree.class.getDeclaredMethod("findParent", BinaryNode.class, Object.class);
		findParent.setAccessible(true);
		rootField = BinaryTree.class.getDeclaredField("root");
		rootField.setAccessible(true);
	}

	/**
	 * Test a tree with no nodes.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws SecurityException
	 * @throws NoSuchMethodException
	 * @throws InvocationTargetException
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 */
	@Test
	public void testEmptyTree() throws NoSuchMethodException, SecurityException, IllegalAccessException,
			IllegalArgumentException, InvocationTargetException,NullPointerException {

		// The next line is equivalent to Object result =
		// tree.findParent(null, "anything");
		Object result = findParent.invoke(tree, null, "anything");

		BinaryNode expected = null;
		assertEquals(expected, result);

	}

	/**
	 * Test a tree with only a root.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws SecurityException
	 * @throws NoSuchFieldException
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 * @throws NoSuchMethodException
	 * @throws InvocationTargetException
	 */
	@Test
	public void testOneNode() throws NoSuchFieldException, SecurityException, IllegalArgumentException,
			IllegalAccessException, NoSuchMethodException, InvocationTargetException {

		// Build the tree and set the root.
		BinaryNode root = new BinaryNode("rootVal");

		// Equivalent to tree.root = root;
		rootField.set(tree, root);

		// Equivalent to tree.findParent(root, "rootVal");
		// We should get back the root.
		Object result = findParent.invoke(tree, root, "rootVal");
		assertEquals(root, result);

		// Equivalent to tree.findParent(root, "anything");
		// We should get back null.
		result = findParent.invoke(tree, root, "anything");
		assertEquals(null, result);

	}

	/**
	 * Test a tree with a root and one left child.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws SecurityException
	 * @throws NoSuchFieldException
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 * @throws NoSuchMethodException
	 * @throws InvocationTargetException
	 */
	@Test
	public void testOneLeftChild() throws NoSuchFieldException, SecurityException, IllegalArgumentException,
			IllegalAccessException, NoSuchMethodException, InvocationTargetException {

		// Build the tree.
		BinaryNode root = new BinaryNode("rootVal");
		BinaryNode left = new BinaryNode("leftVal");
		root.left = left;

		// Equivalent to tree.root = root;
		rootField.set(tree, root);

		// Equivalent to tree.findParent(root, "rootVal");
		// We should get back the root.
		Object result = findParent.invoke(tree, root, "rootVal");
		assertEquals(root, result);

		// Equivalent to tree.findParent(root, "anything");
		// We should get back null.
		result = findParent.invoke(tree, root, "anything");
		assertEquals(null, result);

		// Equivalent to tree.findParent(root, "leftVal");
		// We should get back the left child.
		result = findParent.invoke(tree, root, "leftVal");
		assertEquals(left, result);

	}

	/**
	 * Test a tree with one right child
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws SecurityException
	 * @throws NoSuchFieldException
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 * @throws NoSuchMethodException
	 * @throws InvocationTargetException
	 */
	@Test
	public void testOneRightChild() throws NoSuchFieldException, SecurityException, IllegalArgumentException,
			IllegalAccessException, NoSuchMethodException, InvocationTargetException {

		// Build the tree.
		BinaryNode root = new BinaryNode("rootVal");
		BinaryNode right = new BinaryNode("rightVal");
		root.right = right;

		// Equivalent to tree.root = root;
		rootField.set(tree, root);

		// Equivalent to tree.findParent(root, "rootVal");
		// We should get back the root.
		Object result = findParent.invoke(tree, root, "rootVal");
		assertEquals(root, result);

		// Equivalent to tree.findParent(root, "anything");
		// We should get back null.
		result = findParent.invoke(tree, root, "anything");
		assertEquals(null, result);

		// Equivalent to tree.findParent(root, "rightVal");
		// We should get back the right child.
		result = findParent.invoke(tree, root, "rightVal");
		assertEquals(right, result);
	}

	/**
	 * Test a tree with both children
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws SecurityException
	 * @throws NoSuchFieldException
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 * @throws NoSuchMethodException
	 * @throws InvocationTargetException
	 */
	@Test
	public void testBothChildren() throws NoSuchFieldException, SecurityException, IllegalArgumentException,
			IllegalAccessException, NoSuchMethodException, InvocationTargetException {

		// Build the tree.
		BinaryNode root = new BinaryNode("rootVal");
		BinaryNode left = new BinaryNode("leftVal");
		BinaryNode right = new BinaryNode("rightVal");
		root.left = left;
		root.right = right;

		// Equivalent to tree.root = root;
		rootField.set(tree, root);

		// Equivalent to tree.findParent(root, "rootVal");
		// We should get back the root.
		Object result = findParent.invoke(tree, root, "rootVal");
		assertEquals(root, result);

		// Equivalent to tree.findParent(root, "anything");
		// We should get back null.
		result = findParent.invoke(tree, root, "anything");
		assertEquals(null, result);

		// Equivalent to tree.findParent(root, "leftVal");
		// We should get back the left child.
		result = findParent.invoke(tree, root, "leftVal");
		assertEquals(left, result);

		// Equivalent to tree.findParent(root, "rightVal");
		// We should get back the right child.
		result = findParent.invoke(tree, root, "rightVal");
		assertEquals(right, result);

	}

	/**
	 * Test a tree with several levels.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws SecurityException
	 * @throws NoSuchFieldException
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 * @throws NoSuchMethodException
	 * @throws InvocationTargetException
	 */
	@Test
	public void testSeveralDeep() throws NoSuchFieldException, SecurityException, IllegalArgumentException,
			IllegalAccessException, NoSuchMethodException, InvocationTargetException {

		// Build the tree.
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

		// Equivalent to tree.findParent(root, "rootVal");
		// We should get back the root.
		Object result = findParent.invoke(tree, root, "rootVal");
		assertEquals(root, result);

		// Equivalent to tree.findParent(root, "anything");
		// We should get back null.
		result = findParent.invoke(tree, root, "anything");
		assertEquals(null, result);

		// Equivalent to tree.findParent(root, "left1");
		// We should get back the left child.
		result = findParent.invoke(tree, root, "left1");
		assertEquals(left1, result);

		// Equivalent to tree.findParent(root, "rightVal");
		// We should get back the right child.
		result = findParent.invoke(tree, root, "rightVal");
		assertEquals(right, result);

		// Equivalent to tree.findParent(root, "left2");
		// We should get back left2.
		result = findParent.invoke(tree, root, "left2");
		assertEquals(left2, result);

		// Equivalent to tree.findParent(root, "left3");
		// We should get back left3.
		result = findParent.invoke(tree, root, "left3");
		assertEquals(left3, result);

		// Equivalent to tree.findParent(root, "right4");
		// We should get back the right child.
		result = findParent.invoke(tree, root, "right4");
		assertEquals(right, result);

		// We shouldn't find something higher up the tree.
		// Equivalent to tree.findParent(left2, "left1");
		result = findParent.invoke(tree, left2, "left1");
		assertEquals(null, result);

	}

}
