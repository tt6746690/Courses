package directory_reader;

import java.io.File;
import javax.swing.JFileChooser;

/**
 * Select, read, and print the contents of a directory.
 */
public class DirectoryReader {

	/**
	 * Select a directory, then print the full path to the directory and its
	 * contents, one per line. Prefix the contents with a hyphen and a space.
	 *
	 * @param args
	 *            the command line arguments
	 */
	public static void main(String[] args) {

		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		// commented out because dialogue not showing sometimes
		// http://stackoverflow.com/questions/14640103/jfilechooser-not-showing-up
		// int returnVal = fileChooser.showOpenDialog(null);
		if (fileChooser.showOpenDialog(null) == JFileChooser.APPROVE_OPTION) {
			File file = fileChooser.getSelectedFile();   // getSelectedFile() return File type
			System.out.println(file);
			File[] files = file.listFiles();
			for(File f: files){
				if(f.isFile()){
					System.out.println("- " + f.getName());
				} else if (f.isDirectory()){
					System.out.println("- " + f.getName() + "/");
				}
			}
			
		}
	}
}
