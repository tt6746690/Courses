package photo_renamer;

import java.util.Map;
import java.util.Set;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;

/**
 * The root of a tree representing a directory structure.
 */
public class FileNode {
	/** The name of the file or directory this node represents. */
	private String nom;
	/** Whether this node represents a file or a directory. */
	private FileType type;
	/** This node's parent. */
	private FileNode parent;
	/** The file that this node actually refers to */
	protected File actual_file;
	/**
	 * This node's children, mapped from the file names to the nodes. If type is
	 * FileType.FILE, this is null.
	 */
	private Map<String, FileNode> children;
	private static final String[] image_filetypes = { ".jpg", ".gif", ".png", ".bmp", ".psd", ".tif", ".yuv",
			".pspimage", ".ai", ".ps", ".jpeg", ".tga" };

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
	public FileNode(String nom, FileNode parent, FileType type, File actual_file) {
		this.nom = nom;
		this.parent = parent;
		this.type = type;
		this.actual_file = actual_file;
		this.children = new HashMap<String, FileNode>();
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
		// TODO: complete this method.
		if (this.children.isEmpty() || this.type == FileType.FILE) {
			return null;
		} else {
			FileNode direct_child = this.children.get(name);
			if (direct_child != null) {
				// the case where the node we're looking for is in the level
				// directly below
				return direct_child;
			} else {
				// the case where we search each subtree for the node with name
				Set<String> key_set = this.children.keySet();
				Object[] keys = key_set.toArray();
				for (int i = 0; i < keys.length; i++) {
					FileNode indirect_child = this.children.get(keys[i]).findChild(name);
					if (indirect_child != null) {
						return indirect_child;
					}
				}
				return null;
			}
		}
	}

	/**
	 * Return the name of the file or directory represented by this node.
	 *
	 * @return name of this Node
	 */
	public String getNom() {
		return this.nom;
	}

	/**
	 * Set the name of the current node
	 *
	 * @param name
	 *            of the file/directory
	 */
	public void setName(String nom) {
		this.nom = nom;
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
	 * Build the tree of FileNodes attached from file. Precondition: file must
	 * be a directory
	 */
	public void buildTree(File file) {
		File[] ls = file.listFiles(); // List of files inside file
		for (int i = 0; i < ls.length; i++) { 
			FileType type; 
			if (ls[i].isDirectory()) { // If it's a directory, label it as such
				type = FileType.DIRECTORY;
			} else { // If it isn't, also label it as such
				type = FileType.FILE;
			}
			FileNode child = new FileNode(ls[i].getName(), this, type, ls[i]); 
			this.addChild(ls[i].getName(), child);
			if (child.isDirectory()) {
				child.buildTree(ls[i]); // recursively build tree for the child
			}
		}
	}

	/**
	 * Return a string containing a file's filename extension. Precondition: The
	 * file's filename extension cannot have been artificially removed.
	 **/
	public static String findFileExtension(File file) throws UnexpectedFileTypeException {
		/*
		 * if (file.isDirectory()){ //This code won't work on directories. throw
		 * new UnexpectedFileTypeException("This is a directory"); }
		 */
		int j = file.getName().length() - 1;
		int k = 0;
		String extension = "."; // All filename extensions begin with a .
		boolean stop = false;
		// Does java have a "break" option? I'm not sure.
		// Regardless, stop basically functions like break.
		while (k < file.getName().length() - 1 && !stop) {
			// Iterate over the String representing file's name from the BACK.
			if (file.getName().charAt(j) != '.') {
				// If we don't find a '.', keep going. The filename extensions
				// should begin with a '.'
				k = k + 1;
				j = j - 1; // j is the index of the string being searched. Since
							// we're going backwards, we subtract instead of adding.
			} else {
				// Once the first '.' from the back of the string is found...
				j = j + 1;
				// Increment j by 1, moving to the character immediately after
				// the '.'
				extension = extension + file.getName().substring(j, file.getName().length());
				// Take the substring from the current position, which should
				// the the character after the
				// first '.' to the end of the file. From there, the remaining
				// text in the file's name
				// should be the file extension. Append that to extension.
				stop = true;
				// Break
			}
		}
		return extension;
	}

	public ArrayList<PhotoNode> buildTreeWithImages() throws UnexpectedFileTypeException, IOException {
		if (this.actual_file.isFile()) {
			throw new UnexpectedFileTypeException("This isn't a directory.");
		} else {
			ArrayList<PhotoNode> output = new ArrayList<PhotoNode>();
			File[] ls = this.actual_file.listFiles(); // List of files inside
														// file
			for (int i = 0; i < ls.length; i++) { // A for loop that iterates
													// over ls
				FileType type; // Label the type of each file
				if (ls[i].isDirectory()) { // If it's a directory, label it as
											// such
					type = FileType.DIRECTORY;
				} else { // If it isn't, now we have to check to see if it's an
							// image
					boolean isimage = false;
					String extension = findFileExtension(ls[i]);
					for (String ext : image_filetypes) {
						if ((ext.toUpperCase()).equals(extension.toUpperCase())) {
							isimage = true;
						}
					}
					if (isimage) {
						type = FileType.PHOTO;
						// PhotoNode ph = new PhotoNode(ls[i].getName(), this);
						// output.add(ph);
					} else {
						type = FileType.FILE;
					}
				}
				FileNode child;
				// Build an ordinary FileNode if the file isn't an image file...
				if (type == FileType.FILE || type == FileType.DIRECTORY) {
					child = new FileNode(ls[i].getName(), this, type, ls[i]); // create
																				// a
																				// FileNode
																				// to
																				// represent
																				// the
																				// child
																				// file
																				// in
																				// the
																				// current
																				// directory
				}
				// Or a PhotoNode if it IS an image file.
				else {
					child = new PhotoNode(ls[i].getName(), this, ls[i]); // Or
																			// in
																			// this
																			// case,
																			// a
																			// PhotoNode
					output.add((PhotoNode) child);
				}
				this.addChild(ls[i].getName(), child);
				if (child.isDirectory()) {
					output.addAll(child.buildTreeWithImages()); // recursively
																// build tree
																// for the child
				}
			}
			return output;
		}
	}

	public String openInFinder() {
		return this.parent.actual_file.getAbsolutePath();
	}

	/**
	 * This method is for code that tests this class.
	 * 
	 * @param args
	 *            the command line args.
	 * @throws IOException
	 */
	public ArrayList<PhotoNode> findImagesWithTag(String tagWeAreLookingFor) {
		ArrayList<PhotoNode> output = new ArrayList<PhotoNode>();

		return output;
	}

	public static void main(String[] args) throws UnexpectedFileTypeException, IOException {
		File folder = new File("/Users/anonymous/Desktop/Meep");
		FileNode foldernode = new FileNode("Meep", null, FileType.DIRECTORY, folder);
		ArrayList<PhotoNode> al = foldernode.buildTreeWithImages();
		for (PhotoNode p : al) {
			System.out.println(p.getNom());
		}
		System.out.println(folder.getAbsolutePath());
	}
}
