package photo_renamer;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Collection;

import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JTextArea;

/**
 * The listener for the button to choose a directory. This is where most of the
 * work is done.
 */
public class FileChooserButtonListener implements ActionListener {

	/** The window the button is in. */
	private JFrame directoryFrame;
	/** The label for the full path to the chosen directory. */
	private JLabel directoryLabel;
	/** The file chooser to use when the user clicks. */
	private JFileChooser fileChooser;
	/** The area to use to display the nested directory contents. */
	private PhotoRenamer p;

	/**
	 * An action listener for window dFrame. Changes the directory attribute of the PhotoRenamer class.
	 *
	 * @param dirFrame
	 *            the main window
	 * @param dirLabel
	 *            the label for the directory path
	 * @param fileChooser
	 *            the file chooser to use
	 */
	public FileChooserButtonListener(JFrame dirFrame, JLabel dirLabel, JFileChooser fileChooser, PhotoRenamer p) {
		this.directoryFrame = dirFrame;
		this.directoryLabel = dirLabel;
		this.fileChooser = fileChooser;
		this.p = p;
	}

	/**
	 * Handles the user clicking on the open button.
	 *
	 */
	@Override
	public void actionPerformed(ActionEvent e) {

		int returnVal = fileChooser.showOpenDialog(directoryFrame.getContentPane());

		if (returnVal == JFileChooser.APPROVE_OPTION) {
			File file = fileChooser.getSelectedFile();
			this.p.setDirectory(file);
			System.out.println(file);
			try {
				this.p.buildImageTagger();
			} catch (UnexpectedFileTypeException e1) {
				e1.printStackTrace();
			} catch (IOException e1) {
				e1.printStackTrace();
			}
			this.p.getiFrame().setVisible(true);
			this.p.getdWindow().setVisible(false);
		} else {
			directoryLabel.setText("No Path Selected");
		}
	}

	
}
