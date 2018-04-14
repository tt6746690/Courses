package directory_explorer;

import java.util.Map;
import java.util.Collection;
import java.util.HashMap;

/**
 * The root of a tree representing a directory structure.
 */
public class FileNode {

	/** The name of the file or directory this node represents. */
	private String name;
	/** Whether this node represents a file or a directory. */
	private FileType type;
	/** This node's parent. */
	private FileNode parent;
	/**
	 * This node's children, mapped from the file names to the nodes. If type is
	 * FileType.FILE, this is null.
	 */
	private Map<String, FileNode> children;

	/**
	 * A node in this tree.
	 *
	 * @param name
	 *            the file
	 * @param parent
	 *            the parent node.
	 * @param type
	 *            file or directory
	 * @see buildFileTree
	 */
	public FileNode(String name, FileNode parent, FileType type) {
		this.name = name;
		this.parent = parent;
		this.type = type;
		this.children = new HashMap<String, FileNode>();
		// TODO: complete this method.
	}

	/**
	 * Find and return a child node named name in this directory tree, or null
	 * if there is no such child node.
	 *
	 * @param name
	 *            the file name to search for
	 * @return the node named name
	 */
	public FileNode findChild(String name) {
		FileNode result = null;
		
		if(this.children.containsKey(name)){
			result = this.children.get(name);
		} else {
			Collection<FileNode> allchildren = this.children.values();
			for (FileNode c : allchildren){
				if(c.isDirectory()){
					return c.findChild(name);
				}
			}
		}
		
		return result;
		// TODO: complete this method.
	}

	/**
	 * Return the name of the file or directory represented by this node.
	 *
	 * @return name of this Node
	 */
	public String getName() {
		return this.name;
	}

	/**
	 * Set the name of the current node
	 *
	 * @param name
	 *            of the file/directory
	 */
	public void setName(String name) {
		this.name = name;
	}

	/**
	 * Return the child nodes of this node.
	 *
	 * @return the child nodes directly underneath this node.
	 */
	public Collection<FileNode> getChildren() {
		return this.children.values();
	}

	/**
	 * Return this node's parent.
	 * 
	 * @return the parent
	 */
	public FileNode getParent() {
		return parent;
	}

	/**
	 * Set this node's parent to p.
	 * 
	 * @param p
	 *            the parent to set
	 */
	public void setParent(FileNode p) {
		this.parent = p;
	}

	/**
	 * Add childNode, representing a file or directory named name, as a child of
	 * this node.
	 * 
	 * @param name
	 *            the name of the file or directory
	 * @param childNode
	 *            the node to add as a child
	 */
	public void addChild(String name, FileNode childNode) {
		this.children.put(name, childNode);
	}

	/**
	 * Return whether this node represents a directory.
	 * 
	 * @return whether this node represents a directory.
	 */
	public boolean isDirectory() {
		return this.type == FileType.DIRECTORY;
	}

	/**
	 * This method is for code that tests this class.
	 * 
	 * @param args
	 *            the command line args.
	 */
	public static void main(String[] args) {
		System.out.println("Testing FileNode");
		FileNode f1 = new FileNode("top", null, FileType.DIRECTORY);
		if (!f1.getName().equals("top")) {
			System.out.println("Error: " + f1.getName() + " should be " + "top");
		}
		
		System.out.println("f1's name: " + f1.getName());
		System.out.println("f1's parent: " + f1.getParent());
		
		FileNode c1 = new FileNode("child1", f1, FileType.DIRECTORY);
		FileNode c2 = new FileNode("child2", f1, FileType.FILE);
		
		f1.addChild("child1", c1);
		f1.addChild("child2", c2);
		
		System.out.println("f1's children: " + f1.getChildren());
		f1.setName("toppppp");
		System.out.println("f1's name changed to: " + f1.getName());
				
		System.out.println("c1 is a directory? " + c1.isDirectory());
		System.out.println("c2 is a directory? " + c2.isDirectory()); 
		
		FileNode c3 = new FileNode("child3", c1, FileType.DIRECTORY);
		FileNode c4 = new FileNode("child4", c1, FileType.FILE);
		c1.addChild("child3", c3);
		c1.addChild("child4", c4);
		
		// testing out findChild() 
		System.out.println("testing findChild() method... ");
		System.out.println("try to find child1 in f1 returns: " + f1.findChild("child1").getName());
		System.out.println("try to find child2 in f1 returns: " + f1.findChild("child2").getName());
		System.out.println("try to find child3 in f1 returns: " + f1.findChild("child3").getName());
		System.out.println("try to find child4 in f1 returns: " + f1.findChild("child4").getName());

		
		System.out.println("try to find child3 in c2 returns: " + c1.findChild("child3").getName());
		System.out.println("try to find child4 in c2 returns: " + c1.findChild("child4").getName());


		
	}

}
