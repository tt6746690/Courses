package directory_explorer;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextArea;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class TestFileChooserButtonListener {

	/** The directory in which the test directory structures will be created. */
	private final static String TEST_PATH = ".";

	/** The root of the first test. */
	private String TEST_DIR_1_NAME = TEST_PATH + "/" + "test_dir1";

	/** The root of the second test. */
	private String TEST_DIR_2_NAME = TEST_PATH + "/" + "test_dir2";

	/** The File for the first test directory. */
	private File test1File;

	/** The File for the second test directory. */
	private File test2File;

	/** The window for testing. */
	private JFrame dirFrame = new JFrame();
	/** The label for testing. */
	private JLabel dirLabel = new JLabel("Test");
	/** The text area for testing. */
	private JTextArea textArea = new JTextArea();
	/** The file chooser for testing. */
	private JFileChooser fileChooser = new JFileChooser();
	/** The button listener for testing. */
	private FileChooserButtonListener listener = new FileChooserButtonListener(dirFrame, dirLabel, textArea,
			fileChooser);

	/**
	 * FileNode.buildTree. We'll use reflection to access it because it's
	 * private.
	 */
	private Method buildTree;

	/**
	 * FileNode.buildDirectoryContents. We'll use reflection to access it
	 * because it's private.
	 */
	private Method buildDirectoryContents;

	/**
	 * Create the test directory structures.
	 * @throws IOException 
	 * @throws SecurityException 
	 * @throws NoSuchMethodException 
	 * 
	 * @throws Exception
	 */
	@Before
	public void setUp() throws IOException, NoSuchMethodException, SecurityException {
		// test_dir1/f1
		test1File = new File(TEST_DIR_1_NAME);
		test1File.mkdir(); // test_dir1
		new File(TEST_DIR_1_NAME + "/f1").createNewFile(); // test_dir1/f1

		test2File = new File(TEST_DIR_2_NAME);
		test2File.mkdir(); // test_dir2
		new File(TEST_DIR_2_NAME + "/f1").createNewFile(); // test_dir2/f1
		new File(TEST_DIR_2_NAME + "/z1").createNewFile(); // test_dir2/z1
		new File(TEST_DIR_2_NAME + "/sub1/sub1_1").mkdirs(); // test_dir2/sub1/sub1_1
		new File(TEST_DIR_2_NAME + "/sub1/f2").createNewFile(); // test_dir2/sub1/f2
		new File(TEST_DIR_2_NAME + "/sub1/sub1_1/z2").createNewFile(); // test_dir2/sub1/sub1_1/z2
		new File(TEST_DIR_2_NAME + "/sub2").mkdirs(); // test_dir2/sub2
		new File(TEST_DIR_2_NAME + "/sub2/z1").createNewFile(); // test_dir2/sub2/z1

		// Make the private methods accessible.
		buildTree = FileChooserButtonListener.class.getDeclaredMethod("buildTree", File.class, FileNode.class);
		buildTree.setAccessible(true);
		buildDirectoryContents = FileChooserButtonListener.class.getDeclaredMethod("buildDirectoryContents",
				FileNode.class, StringBuffer.class, String.class);
		buildDirectoryContents.setAccessible(true);
	}

	/**
	 * Delete the test directories.
	 * 
	 * @throws IOException
	 */
	@After
	public void tearDown() throws IOException {
		delete(new File(TEST_DIR_1_NAME));
		delete(new File(TEST_DIR_2_NAME));
	}

	/**
	 * Delete f and its contents.
	 * 
	 * @param f
	 *            the file to delete.
	 */
	private static void delete(File f) {
		if (f.isDirectory()) {
			for (File c : f.listFiles())
				delete(c);
		}
	}

	/**
	 * Test getChildren on the first test directory.
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
	public void testBuildTreeDir1() throws NoSuchMethodException, SecurityException, IllegalAccessException,
			IllegalArgumentException, InvocationTargetException {

		FileNode root = new FileNode(test1File.getName(), null, FileType.DIRECTORY);

		buildTree.invoke(listener, test1File, root);

		assertEquals("test_dir1", root.getName());
		assertEquals(null, root.getParent());
		assertTrue(root.isDirectory());

		// There should be one child File, f1.
		Collection<FileNode> children = root.getChildren();
		assertEquals(1, children.size());
		for (FileNode child : children) {
			assertEquals("f1", child.getName());
			assertEquals(root, child.getParent());
			assertFalse(child.isDirectory());
		}

	}

	/**
	 * Test getChildren on the second test directory.
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
	public void testBuildTreeDir2() throws NoSuchMethodException, SecurityException, IllegalAccessException,
			IllegalArgumentException, InvocationTargetException {
		FileNode root = new FileNode(test2File.getName(), null, FileType.DIRECTORY);

		buildTree.invoke(listener, test2File, root);

		// Check the root.
		assertEquals("test_dir2", root.getName());
		assertEquals(null, root.getParent());
		assertTrue(root.isDirectory());

		// test_dir2
		// --f1
		// --sub1
		// ----f2
		// ----sub1_1
		// ------z1
		// --sub2
		// --z1 <-- We're testing with a filename that is used twice.

		// There should be 4 children of test_dir2: f1, sub1, sub2, and z1.
		Collection<FileNode> children = root.getChildren();
		List<String> rootChildrenNames = new ArrayList<String>(Arrays.asList("f1", "sub1", "sub2", "z1"));
		assertEquals(4, children.size());
		
		for (FileNode child : children) {
			assertTrue(rootChildrenNames.contains(child.getName()));
			assertEquals(root, child.getParent());

			if ("f1".equals(child.getName()) || "z1".equals(child.getName())) {
				assertFalse(child.isDirectory());
			} else if ("sub1".equals(child.getName())) {
				assertTrue(child.isDirectory());

				// test_dir2/sub1 has two children: f2 and sub1_1.
				Collection<FileNode> sub1Children = child.getChildren();
				List<String> sub1Names = new ArrayList<String>(Arrays.asList("f2", "sub1_1"));
				assertEquals(2, sub1Children.size());
				for (FileNode sub1Child : sub1Children) {
					assertEquals(child, sub1Child.getParent());
					assertTrue(sub1Names.contains(sub1Child.getName()));
					if ("f2".equals(sub1Child.getName())) {
						assertEquals(child, sub1Child.getParent());
						assertFalse(sub1Child.isDirectory());
					} else if ("sub1_1".equals(sub1Child.getName())) {
						assertEquals(child, sub1Child.getParent());
						assertTrue(sub1Child.isDirectory());
						// test_dir2/sub1/sub1_1 has one child: z2.
						Collection<FileNode> sub1_1Children = sub1Child.getChildren();
						assertEquals(1, sub1_1Children.size());
						for (FileNode sub1_1Child : sub1_1Children) {
							assertEquals("z2", sub1_1Child.getName());
							assertEquals(sub1Child, sub1_1Child.getParent());
							assertFalse(sub1_1Child.isDirectory());
						}
					}
				}
			} else if ("sub2".equals(child.getName())) {
				assertEquals(root, child.getParent());
				assertTrue(child.isDirectory());

				// test_dir2/sub2 has one child: z1.
				Collection<FileNode> sub2Children = child.getChildren();
				assertEquals(1, sub2Children.size());
				for (FileNode sub2Child : sub2Children) {
					assertEquals("z1", sub2Child.getName());
					assertEquals(child, sub2Child.getParent());
					assertFalse(sub2Child.isDirectory());
				}
			}
		}
	}

	/**
	 * Test buildDirectoryContents with nesting 5 levels deep.
	 * 
	 * @throws InvocationTargetException
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 */
	@Test
	public void testBuildDirectoryContentsStraightPath()
			throws IllegalAccessException, IllegalArgumentException, InvocationTargetException {
		// root
		// --child0
		// ----child1
		// ------child2
		// --------child3
		FileNode root = new FileNode("root", null, FileType.DIRECTORY);
		FileNode child0 = new FileNode("child0", root, FileType.DIRECTORY);
		FileNode child1 = new FileNode("child1", child0, FileType.DIRECTORY);
		FileNode child2 = new FileNode("child2", child1, FileType.DIRECTORY);
		FileNode child3 = new FileNode("child3", child2, FileType.FILE);

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
			root.addChild(child0.getName(), child0);
			child0.addChild(child1.getName(), child1);
			child1.addChild(child2.getName(), child2);
			child2.addChild(child3.getName(), child3);
		}

		StringBuffer buffer = new StringBuffer();
		buildDirectoryContents.invoke(listener, root, buffer, "");
		String res = "root\n--child0\n----child1\n------child2\n--------child3";
		String buf = buffer.toString();
		buf = trimTrailingNewline(buf);
		assertEquals(res, buf);
	}

	/**
	 * Test buildDirectoryContents with two branches at the first level and 5
	 * levels deep.
	 * 
	 * @throws InvocationTargetException
	 * @throws IllegalArgumentException
	 * @throws IllegalAccessException
	 */
	@Test
	public void testBuildDirectoryContentsBranching()
			throws IllegalAccessException, IllegalArgumentException, InvocationTargetException {

		// root
		// --child0a
		// ----child1
		// ------child2
		// --------child3
		// --child0b
		FileNode root = new FileNode("root", null, FileType.DIRECTORY);
		FileNode child0b = new FileNode("child0b", root, FileType.DIRECTORY);
		FileNode child0a = new FileNode("child0a", root, FileType.DIRECTORY);
		FileNode child1 = new FileNode("child1", child0a, FileType.DIRECTORY);
		FileNode child2 = new FileNode("child2", child1, FileType.DIRECTORY);
		FileNode child3 = new FileNode("child3", child2, FileType.FILE);

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
			root.addChild(child0a.getName(), child0a);
			root.addChild(child0b.getName(), child0b);
			child0a.addChild(child1.getName(), child1);
			child1.addChild(child2.getName(), child2);
			child2.addChild(child3.getName(), child3);
		}

		StringBuffer buffer = new StringBuffer();
		buildDirectoryContents.invoke(listener, root, buffer, "");

		// We're not sure which order the directories will be traversed in. Here
		// are the possible results.
		String result1 = "root\n--child0b\n--child0a\n----child1\n------child2\n--------child3";
		String result2 = "root\n--child0a\n----child1\n------child2\n--------child3\n--child0b";
		String buf = buffer.toString();
		buf = trimTrailingNewline(buf);
		assertTrue(result1.equals(buf) || result2.equals(buf));
	}

	/**
	 * Delete a trailing \n if it exists.
	 * @param buf the String to trim.
	 * @return a copy of buf but with a trailing newline removed
	 */
	private String trimTrailingNewline(String buf) {
		if (buf.endsWith("\n")) {
			buf = buf.substring(0, buf.length() - 1);
		}
		return buf;
	}
}
