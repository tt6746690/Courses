package photo_renamer;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;

public class PhotoNode extends FileNode {

	private ArrayList<String> tags; // A list of tags that are possessed by each
									// photo.
	public static ArrayList<String> allTags = new ArrayList<String>();;

	public PhotoNode(String name, FileNode parent, File file) throws IOException {
		super(name, parent, FileType.PHOTO, file);
		this.tags = new ArrayList<String>();
	}

	/**
	 * Returns true iff string newTag is a valid tag. A valid tag begins with an
	 * '@' and does not contain any whitespace, or any '@'s other than the first
	 * '@' at the beginning.
	 * 
	 * @param newTag
	 *            The string we are checking to see if it is a valid tag.
	 * @return Whether or not newTag is a valid tag.
	 */
	public static boolean isValidTag(String newTag) {
		if (newTag.charAt(0) != '@') {
			return false;
		}
		for (int i = 1; i < newTag.length(); i++) {
			if (newTag.charAt(i) == '@' || newTag.charAt(i) == ' ') {
				return false;
			}
		}
		return true;
	}

	/**
	 * Adds a tag to the current instance of a photoNode that represents an
	 * image, assuming it meets the following conditions. First, that the newTag
	 * is already in the master list of tags. Second, the tag must be a valid
	 * tag. A valid tag is a tag that begins with an
	 * 
	 * @, and does not contain any whitespace, nor any other @s other than the
	 *    original one.
	 * 
	 * @param newTag
	 *            the tag to be added to be the photo.
	 */
	public void addTagToPhoto(String newTag) {
		if (allTags.contains(newTag)) {
			if (isValidTag(newTag) && !this.tags.contains(newTag)) {
				this.tags.add(newTag);
			} else {
				System.out.println("Invalid Tag! Either this is not a valid tag, or this photo already has this tag.");
			}
		} else {
			System.out.println("This tag does not currently exist."
					+ " We implore you to create it using the new tag function first.");
		}
	}

	/**
	 * Remove a tag from the current instance of photoNode that represents an
	 * image.
	 * 
	 * @param toDelete
	 *            the tag to be deleted.
	 */
	public void deleteTagFromPhoto(String toDelete) {
		this.tags.remove(toDelete);
	}

	/**
	 * Return a String of a filename, with all the tags removed.
	 * 
	 * @param name
	 *            The filename to be stripped of its tags.
	 * @return The filename without its tags.
	 */
	public static String stripOfTags(String name) {
		String[] s = name.split(" ");
		return s[s.length - 1];
	}

	/**
	 * After the user has added and removed tags from the photo, this method
	 * actually renames the photo in the file system, and commits the changes to
	 * the log.
	 * 
	 * @throws URISyntaxException
	 * @throws IOException
	 * 
	 * @return The new name of the file.
	 */
	public String confirmRename() {
		// Build the desired new name
		String taggedName = "";
		if (!this.getTags().isEmpty()) {
			for (String t : this.getTags()) {
				taggedName = taggedName + t + " ";
			}
			taggedName = taggedName.substring(0, taggedName.length() - 1);
		}
		String newname1 = this.actual_file.getParent() + "/" + taggedName + " "
				+ stripOfTags(this.actual_file.getName());

		// Build the new file.
		File newname = new File(newname1);

		// Rename the file.
		if (this.actual_file.renameTo(newname)) {
			System.out.println("yay");
			this.actual_file = newname;
			this.setName(taggedName + " " + stripOfTags(this.actual_file.getName()));
		} else {
			System.out.println("aww");
		}
		return taggedName + " " + stripOfTags(this.actual_file.getName());
	}

	/**
	 * After the user has added and removed tags from the photo, this method
	 * actually renames the photo in the file system, and commits the changes to
	 * the log.
	 * 
	 * @param renameTo
	 *            The desired new name of the file.
	 */
	public void confirmRename(String renameTo) {
		String newname1 = this.actual_file.getParent() + "/" + renameTo;
		String newname2 = renameTo;
		File newname = new File(newname1);
		if (this.actual_file.renameTo(newname)) {
			System.out.println("yay");
			this.actual_file = newname;
			this.setName(renameTo);
		} else {
			System.out.println("aww");
		}
	}

	/**
	 * Given a string s, and a position in the string, i, return the substring
	 * of s beginning at the position i and ending at the next whitespace. If
	 * there is no more whitespace after index i, return the substring beginning
	 * at i and ending at the end of s
	 * 
	 * @param s
	 *            The string we want to search through.
	 * @param i
	 *            The position of s we start at.
	 * @return
	 */
	public static String untilNextWhitespace(String s, int i) {
		String output = "";
		for (int j = i; j < s.length(); j++) {
			if (s.charAt(j) != ' ') {
				output = output + s.charAt(j);
			} else {
				break;
			}
		}
		return output;
	}

	/**
	 * Reads the filename of an image of the current instance of the photoNode,
	 * and then determines what tags it has. Afterwards, adds them to the
	 * PhotoNode's tags.
	 * 
	 */
	public void setTags() {
		ArrayList<String> tags = new ArrayList<String>();
		String fname = this.actual_file.getName();
		for (int i = 0; i < fname.length(); i++) {
			if (fname.charAt(i) == '@') {
				tags.add(untilNextWhitespace(fname, i));
			}
		}
		this.tags = tags;
	}

	/**
	 * Getter function for the tags of a PhotoNode.
	 * 
	 * @return The tags of a PhotoNode
	 */
	public ArrayList<String> getTags() {
		return this.tags;
	}

	/**
	 * Read the file that keeps track of the tags and update the master list of
	 * tags
	 * 
	 * @throws IOException
	 */
	public static void initializeTags() throws IOException {
		File tagsFile = new File("photo_renamer/tags.txt");
		tagsFile.mkdirs();
		tagsFile.mkdirs();
		if (!tagsFile.exists()) {
			tagsFile.createNewFile();
		}
		try {
			String nextLine;
			BufferedReader br = new BufferedReader(new FileReader("photo_renamer/tags.txt"));
			while ((nextLine = br.readLine()) != null) {
				String nextTag = nextLine.trim();
				if (!allTags.contains(nextTag)) {
					allTags.add(nextTag);
				}
			}
			br.close();
		} catch (FileNotFoundException e0) {
			String nextLine;
			BufferedReader br = new BufferedReader(new FileReader("src/photo_renamer/tags.txt"));
			while ((nextLine = br.readLine()) != null) {
				String nextTag = nextLine.trim();
				if (!allTags.contains(nextTag)) {
					allTags.add(nextTag);
				}
			}
			br.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	/**
	 * Create a new tag, adding it to the master list of tags.
	 * 
	 * @param tagName
	 *            The tag to be added. Tags must be a valid tag. Valid tags must
	 *            begin with an @ and contain no whitespace nor any other @s.
	 * @return Return true if the tag is created successfully, and false if it
	 *         is not.
	 * @throws IOException
	 */
	public static boolean createNewTag(String tagName) throws IOException {
		if (isValidTag(tagName)) {
			if (!allTags.contains(tagName)) {
				// Add tagName to the master list of tags in the PhotoNode
				allTags.add(tagName);

				// Update the text file containing the tags.
				File tagsFile = new File("photo_renamer/tags.txt");
				tagsFile.mkdirs();
				try {
					FileWriter fw = new FileWriter(tagsFile.getAbsoluteFile());
					BufferedWriter bw = new BufferedWriter(fw);
					for (int i = 0; i < allTags.size(); i++) {
						bw.write(allTags.get(i));
						bw.write("\n");
					}
					bw.close();
				} catch (FileNotFoundException f) {
					File tagsFile1 = new File("src/photo_renamer/tags.txt");
					FileWriter fw = new FileWriter(tagsFile1.getAbsoluteFile());
					BufferedWriter bw = new BufferedWriter(fw);
					for (int i = 0; i < allTags.size(); i++) {
						bw.write(allTags.get(i));
						bw.write("\n");
					}
					bw.close();
				}
				return true;
			} else {
				return false;
			}
		} else {
			System.out.println(
					"This is not a valid tag. Recall that a valid tag begins with an '@' and does not contain any whitespace"
							+ "or any '@'s other than the one at the beginning.");
			return false;
		}
	}

	/**
	 * Remove a tag from all the the master list of tags
	 * 
	 * @param tagName
	 *            The tag to be removed.
	 * @return Return true if the tag is successfully removed, and false
	 *         otherwise.
	 * @throws IOException
	 */
	public static boolean deleteTag(String tagName) throws IOException {
		PhotoRenamer p = PhotoRenamer.getInstanceOf();
		if (!p.usingTag(tagName)) {
			// Remove the tag from the PhotoNodes
			allTags.remove(tagName);

			// Edit the text file to remove the tagName
			File tagsFile = new File("photo_renamer/tags.txt");
			tagsFile.mkdirs();
			try {
				FileWriter fw = new FileWriter(tagsFile.getAbsoluteFile());
				BufferedWriter bw = new BufferedWriter(fw);
				for (int i = 0; i < allTags.size(); i++) {
					bw.write(allTags.get(i));
					bw.write("\n");
				}
				bw.close();
			} catch (FileNotFoundException e) {
				File tagsFile1 = new File("src/photo_renamer/tags.txt");
				tagsFile1.mkdirs();
				FileWriter fw = new FileWriter(tagsFile1.getAbsoluteFile());
				BufferedWriter bw = new BufferedWriter(fw);
				for (int i = 0; i < allTags.size(); i++) {
					bw.write(allTags.get(i));
					bw.write("\n");
				}
				bw.close();
			}
			return true;
		} else {
			System.out.println("Cannot delete this tag: some photos are using it!");
			return false;
		}
	}

	/**
	 * Returns an ArrayList of file names that contain a tag.
	 * 
	 * @param al
	 * @param tagName
	 * @return
	 */
	public static ArrayList<String> photosWithTag(ArrayList<PhotoNode> al, String tagName) {
		ArrayList<String> output = new ArrayList<String>();
		for (PhotoNode p : al) {
			if (p.getTags().contains(tagName)) {
				output.add(p.getNom());
			}
		}
		return output;
	}

	/**
	 * Find the tag that features the most in the list of the ArrayList of
	 * PhotoNodes al.
	 * 
	 * @param al
	 *            The list of PhotoNodes to search through.
	 * @return The most common tag.
	 */
	public static String mostCommonTag(ArrayList<PhotoNode> al) {
		// Create an array the same size of all the tags
		int n = PhotoNode.allTags.size();
		int[] tags = new int[n];

		// Iterate through all the PhotoNodes in al, and increment each time a
		// tag is found.
		for (PhotoNode p : al) {
			for (String t : p.tags) {
				int m = PhotoNode.allTags.indexOf(t);
				tags[m] = tags[m] + 1;
			}
		}

		// Return the largest tag.
		int j = 0;
		for (int i = 0; i < tags.length; i++) {
			if (tags[i] > tags[j]) {
				j = i;
			}
		}
		return PhotoNode.allTags.get(j);
	}

	/**
	 * Searches through an ArrayList of PhotoNodes and returns the PhotoNode
	 * with the greatest number of tags.
	 * 
	 * @param al
	 *            The ArrayList we are searching through
	 * @return The PhotoNode with the greatest number of tags.
	 */
	public static PhotoNode mostTags(ArrayList<PhotoNode> al) {
		PhotoNode most = al.get(0);
		for (PhotoNode p : al) {
			if (p.tags.size() > most.tags.size()) {
				most = p;
			}
		}
		return most;
	}

	/**
	 * toString method. Just returns the filename.
	 * 
	 */
	public String toString() {
		return this.actual_file.getName();
	}

	public static void main(String[] args) throws IOException {
		String name = "@Toronto @It_works @Rabbit bunny.png";
		System.out.println(stripOfTags(name));

	}

}
