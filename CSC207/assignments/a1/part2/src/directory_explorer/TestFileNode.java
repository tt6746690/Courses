/**
 * 
 */
package directory_explorer;

import static org.junit.Assert.*;

import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.HashMap;
import java.util.Map;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * @author pgries
 *
 */
public class TestFileNode {

	/**
	 * Test a single File.DIRECTORY node.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws SecurityException
	 * @throws NoSuchFieldException
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 */
	@Test
	public void testConstructorOneDirectory()
			throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		FileNode root = new FileNode("root", null, FileType.DIRECTORY);
		Field name = root.getClass().getDeclaredField("name");
		Field type = root.getClass().getDeclaredField("type");
		Field parent = root.getClass().getDeclaredField("parent");
		Field children = root.getClass().getDeclaredField("children");
		name.setAccessible(true);
		type.setAccessible(true);
		parent.setAccessible(true);
		children.setAccessible(true);

		assertEquals("root", name.get(root));
		assertEquals(FileType.DIRECTORY, type.get(root));
		assertEquals(null, parent.get(root));

		Map<String, FileNode> expected = new HashMap<String, FileNode>();
		assertEquals(expected, children.get(root));
	}

	/**
	 * Test whether a single File.FILE node's children is null, as specified by
	 * its Javadoc.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws SecurityException
	 * @throws NoSuchFieldException
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 */
	@Test
	public void testConstructorOneFile()
			throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {
		FileNode root = new FileNode("root", null, FileType.DIRECTORY);
		FileNode child = new FileNode("a child", root, FileType.FILE);
		Field children = child.getClass().getDeclaredField("children");
		children.setAccessible(true);
		assertEquals(null, children.get(child));
	}

	/**
	 * Test a node with a single child.
	 * 
	 * We don't have a way to fix reflection exceptions, so we propagate.
	 * 
	 * @throws SecurityException
	 * @throws NoSuchFieldException
	 * @throws IllegalAccessException
	 * @throws IllegalArgumentException
	 */
	@Test
	public void testConstructorOneChild()
			throws NoSuchFieldException, SecurityException, IllegalArgumentException, IllegalAccessException {

		FileNode root = new FileNode("root", null, FileType.DIRECTORY);
		FileNode child = new FileNode("a child", root, FileType.FILE);

		// Some solutions added the child to the root's children map in the
		// constructor. If they didn't, then we need to call addChild.

		// JGCHANGE : Check to make sure children is initialized
		boolean childrenNull = false;
		try
		{
			root.getChildren();
		}
		catch(NullPointerException e)
		{
			childrenNull  = true;
		}
		assertEquals(false, childrenNull);

		
		if (root.getChildren().size() == 0) {
			root.addChild(child.getName(), child);
		}
		
		Field name = root.getClass().getDeclaredField("name");
		Field type = root.getClass().getDeclaredField("type");
		Field parent = root.getClass().getDeclaredField("parent");
		Field children = root.getClass().getDeclaredField("children");
		name.setAccessible(true);
		type.setAccessible(true);
		parent.setAccessible(true);
		children.setAccessible(true);

		// Check the root.
		assertEquals("root", name.get(root));
		assertEquals(FileType.DIRECTORY, type.get(root));
		assertEquals(null, parent.get(root));

		Map<String, FileNode> expected = new HashMap<String, FileNode>();
		expected.put("a child", child);
		Map<String, FileNode> result = (Map<String, FileNode>) children.get(root);
		assertEquals(expected, result);

		// Check the name and type of the child.
		assertEquals("a child", name.get(child));
		assertEquals(FileType.FILE, type.get(child));
		assertEquals(root, parent.get(child));
	}

	/**
	 * Test finding the first child of the root node.
	 */
	@Test
	public void testFindChildRoot() {

		FileNode root = new FileNode("root", null, FileType.DIRECTORY);
		FileNode child = new FileNode("a child", root, FileType.FILE);

		// Some solutions added the child to the root's children map in the
		// constructor. If they didn't, then we need to call addChild.

		// JGCHANGE
		boolean childrenNull = false;
		try
		{
			root.getChildren();
		}
		catch(NullPointerException e)
		{
			childrenNull  = true;
		}
		assertEquals(false, childrenNull);		
		
		if (root.getChildren().size() == 0) {
			root.addChild(child.getName(), child);
		}

		FileNode res = root.findChild("a child");
		assertEquals(child, res);
	}

	/**
	 * Test finding the second child of the root node.
	 */
	@Test
	public void testFindChildTwoChildren() {

		FileNode root = new FileNode("root", null, FileType.DIRECTORY);
		FileNode child1 = new FileNode("child1", root, FileType.FILE);
		FileNode child2 = new FileNode("child2", root, FileType.FILE);

		// Some solutions added the child to the root's children map in the
		// constructor. If they didn't, then we need to call addChild.
		// JGCHANGE
		boolean childrenNull = false;
		try
		{
			root.getChildren();
		}
		catch(NullPointerException e)
		{
			childrenNull  = true;
		}
		assertEquals(false, childrenNull);
		
		if (root.getChildren().size() == 0) {
			root.addChild(child1.getName(), child1);
			root.addChild(child2.getName(), child2);
		}

		FileNode res = root.findChild("child2");
		assertEquals(child2, res);
	}

	/**
	 * Test finding a deeper child of the root node where the target is the
	 * child of the first child of the root.
	 */
	@Test
	public void testFindChildTwoDeepFirstChild() {

		FileNode root = new FileNode("root", null, FileType.DIRECTORY);
		FileNode child1 = new FileNode("child1", root, FileType.DIRECTORY);
		FileNode child2 = new FileNode("child2", root, FileType.DIRECTORY);
		FileNode child1a = new FileNode("child1a", child1, FileType.FILE);

		// Some solutions added the child to the root's children map in the
		// constructor. If they didn't, then we need to call addChild.
		
		// JGCHANGE : Check to make sure children is initialized

		boolean childrenNull = false;
		try
		{
			root.getChildren();
		}
		catch(NullPointerException e)
		{
			childrenNull  = true;
		}
		assertEquals(false, childrenNull);
		
		
		if (root.getChildren().size() == 0) {
			root.addChild(child1.getName(), child1);
			root.addChild(child2.getName(), child2);
			child1.addChild(child1a.getName(), child1a);
		}

		FileNode res = root.findChild("child1a");
		assertEquals(child1a, res);
	}

	/**
	 * Test finding a deeper child of the root node where the target is the
	 * child of the second child of the root.
	 */
	@Test
	public void testFindChildTwoDeepSecondChild() {

		FileNode root = new FileNode("root", null, FileType.DIRECTORY);
		FileNode child1 = new FileNode("child1", root, FileType.DIRECTORY);
		FileNode child2 = new FileNode("child2", root, FileType.DIRECTORY);
		FileNode child2a = new FileNode("child2a", child2, FileType.FILE);

		// Some solutions added the child to the root's children map in the
		// constructor. If they didn't, then we need to call addChild.
		// JGCHANGE
		boolean childrenNull = false;
		try
		{
			root.getChildren();
		}
		catch(NullPointerException e)
		{
			childrenNull  = true;
		}
		assertEquals(false, childrenNull);		
		
		if (root.getChildren().size() == 0) {
			root.addChild(child1.getName(), child1);
			root.addChild(child2.getName(), child2);
			child2.addChild(child2a.getName(), child2a);
		}

		FileNode res = root.findChild("child2a");
		assertEquals(child2a, res);
	}

}
