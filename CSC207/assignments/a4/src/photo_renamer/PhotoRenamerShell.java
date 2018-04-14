package photo_renamer;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.net.URISyntaxException;
import java.util.*;

public class PhotoRenamerShell {

	private File file;
	public static ArrayList<LogEntry> log = new ArrayList<LogEntry>();

	public PhotoRenamerShell(File file) {
		this.file = file;
	}
	
	public File getFile(){
		return this.file;
	}
	
	public ArrayList<PhotoNode> readySet() throws UnexpectedFileTypeException, IOException{
		FileNode fileTree = new FileNode(this.file.getName(), null, FileType.DIRECTORY, this.file);
		ArrayList<PhotoNode> photos = fileTree.buildTreeWithImages();
		buildLog(photos);
		PhotoNode.initializeTags();

		for (int i = 0; i < photos.size(); i++) {
			photos.get(i).setTags();
			ArrayList<String> thisPhotoTags = photos.get(i).getTags();
			for (int j = 0; j < thisPhotoTags.size(); j++) {
				if (!PhotoNode.allTags.contains(thisPhotoTags.get(j))) {
					PhotoNode.allTags.add(thisPhotoTags.get(j));
				}
			}
		}
		return photos;
	}

	// Alternate version of ready_set() that takes in a parameter
	// For use in PhotoRenamer
	public static ArrayList<PhotoNode> readySet(File f) throws UnexpectedFileTypeException, IOException{
		FileNode fileTree = new FileNode(f.getName(), null, FileType.DIRECTORY, f);
		ArrayList<PhotoNode> photos = fileTree.buildTreeWithImages();
		buildLog(photos);
		PhotoNode.initializeTags();

		for (int i = 0; i < photos.size(); i++) {
			photos.get(i).setTags();
			ArrayList<String> thisPhotoTags = photos.get(i).getTags();
			for (int j = 0; j < thisPhotoTags.size(); j++) {
				if (!PhotoNode.allTags.contains(thisPhotoTags.get(j))) {
					PhotoNode.allTags.add(thisPhotoTags.get(j));
				}
			}
		}
		return photos;
	}
	
	public void run() throws IOException, UnexpectedFileTypeException, URISyntaxException {
		if (this.file.isDirectory()) {
			Scanner s = new Scanner(System.in);
			ArrayList<PhotoNode> photos = readySet();

			// traverse through all the PhotoNodes
			for (int i = 0; i < photos.size(); i++) {
				PhotoNode currentPhoto = photos.get(i);
				System.out.println("The current photo is " + currentPhoto.getNom());
				// display all the tags the photo currently has
				System.out.println("The current photo has the following tags: ");
				System.out.println(currentPhoto.getTags());

				// display the master list of existing tags
				System.out.println("Choose from this master list of tags, or create new ones: ");
				PhotoNode.initializeTags();
				System.out.println(PhotoNode.allTags);

				boolean moreOptions = true;
				while (moreOptions) {
					System.out.println("Choose an option: add tag, remove tag, new tag, "
							+ "delete tag, see name without tags, see log, confirm rename, revert to, \n "
							+ "next photo, previous photo, open in finder, find photos with tag, most common tag, most tags,  \n"
							+ "most similar photos");
					String option = s.nextLine();
					System.out.println("You chose: " + option);
					String tagName;
					switch (option) {
					case "add tag":
						System.out.println("Enter tag name to add: ");
						tagName = s.nextLine();
						currentPhoto.addTagToPhoto(tagName);
						System.out.println("The current photo now has the following tags: ");
						System.out.println(currentPhoto.getTags());
						break;
					case "remove tag":
						System.out.println("Enter tag name to remove: ");
						tagName = s.nextLine();
						currentPhoto.deleteTagFromPhoto(tagName);
						System.out.println("The current photo now has the following tags: ");
						System.out.println(currentPhoto.getTags());
						break;
					case "new tag":
						System.out.println("Enter new tag name: ");
						tagName = s.nextLine();
						PhotoNode.createNewTag(tagName);
						System.out.println("The current master list of tags is");
						System.out.println(PhotoNode.allTags);
						break;
					case "delete tag":
						System.out.println("Enter tag name to delete: ");
						tagName = s.nextLine();
						PhotoNode.deleteTag(tagName);
						System.out.println("The current master list of tags is");
						System.out.println(PhotoNode.allTags);
						break;
					case "see log":
						// display the log
						this.displayLog(currentPhoto.actual_file.getName());
						break;
					case "confirm rename":
						String oldName = currentPhoto.actual_file.getName();
						String newName = currentPhoto.confirmRename();
						LogEntry l = getLog(oldName);
						l.newName(newName);
						System.out.println("Your renaming is confirmed! Use \"next photo\" to move on.");
						break;
					case "revert to":
						this.displayLog(currentPhoto.actual_file.getName());
						System.out.println("Enter the index of the name you want to revert to: ");
						String input = s.nextLine();
						int index = Integer.parseInt(input);
						LogEntry m = this.getLog(currentPhoto.actual_file.getName());
						String revertedName = m.getRevertedName(index);
						System.out.println(revertedName);
						currentPhoto.confirmRename(revertedName);
						m.newName(revertedName);
						break;
					case "next photo":
						System.out.println("Moving on to next photo...");
						moreOptions = false;
						break;
					case "previous photo":
						System.out.print("Returning to the previous photo...");
						moreOptions = false;
						i -= 2;
						break;
					case "open in finder":
						System.out.println(currentPhoto.openInFinder());
						break;
					case "find photos with tag":
						System.out.println("Enter a tag to search images for");
						tagName = s.nextLine();
						if (!PhotoNode.allTags.contains(tagName)) {
							System.out.println("Invalid Tag!");
						} else {
							System.out.println(PhotoNode.photosWithTag(photos, tagName));
						}
						break;
					case "most common tag":
						System.out.println(PhotoNode.mostCommonTag(photos));
						break;
					case "most tags":
						System.out.println(PhotoNode.mostTags(photos).getNom());
						break;
					case "see name without tags":
						System.out.println(PhotoNode.stripOfTags(currentPhoto.getNom()));
						break;
					default:
						System.out.println("Please enter a valid option!");
					}
					serializeLog();
				}
			}
			s.close();
		} else

		{

		}
	}

	public static void buildLog(ArrayList<PhotoNode> photos) {
		File logFile = new File("src/photo_renamer/log.ser");
		if (logFile.exists()) {
			deserializeLog();
		}
		// create LogEntry for each PhotoNode
		for (int i = 0; i < photos.size(); i++) {
			LogEntry l = new LogEntry(photos.get(i).getNom());
			if (!log.contains(l)) {
				log.add(l);
			}
		}
	}

	public static void deserializeLog() {
		try {
			FileInputStream fi = new FileInputStream("src/photo_renamer/log.ser");
			ObjectInputStream in = new ObjectInputStream(fi);
			log = (ArrayList<LogEntry>) in.readObject();
			in.close();
			fi.close();
		} catch (IOException i) {
			i.printStackTrace();
		} catch (ClassNotFoundException c) {
			c.printStackTrace();
		}
	}

	public static void serializeLog() {
		try {
			FileOutputStream fileOut = new FileOutputStream("src/photo_renamer/log.ser");
			ObjectOutputStream out = new ObjectOutputStream(fileOut);
			out.writeObject(log);
			out.close();
			fileOut.close();
		} catch (IOException i) {
			i.printStackTrace();
		}
	}

	public void displayLog(String currName) {
		System.out.println(this.getLog(currName).toString());
	}

	// find the correct LogEntry object in this.log
	public static LogEntry getLog(String currName) {
		for (int j = 0; j < log.size(); j++) {
			if (log.get(j).getCurrName().equals(currName)) {
				return log.get(j);
			}
		}
		return null;
	}

	public static void main(String[] args) throws UnexpectedFileTypeException, URISyntaxException, IOException {
		
		File file = new File("/Users/anonymous/Desktop/photos");
		PhotoRenamerShell shell = new PhotoRenamerShell(file);
		try {
			shell.run();
		} catch (IOException e1) {
			System.out.println("IO exception!");
		} 
		/*
		PhotoNode.initializeTags();
		for(String t: PhotoNode.allTags){
			System.out.println(t);
		}
		*/
	}
}
