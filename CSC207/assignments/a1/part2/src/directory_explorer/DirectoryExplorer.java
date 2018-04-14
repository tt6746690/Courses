package directory_explorer;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;

import javax.swing.AbstractButton;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

/**
 * Create and show a directory explorer, which displays the contents of a
 * directory.
 */
public class DirectoryExplorer {

	/**
	 * The prefix to use when displaying nested files and directories; each
	 * nested level has one more copy of this.
	 */
	public final static String PREFIX = "--";

	/**
	 * Create and return the window for the directory explorer.
	 *
	 * @return the window for the directory explorer
	 */
	public static JFrame buildWindow() {
		JFrame directoryFrame = new JFrame("Directory Explorer");

		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

		JLabel directoryLabel = new JLabel("Select a directory");

		// Set up the area for the directory contents.
		JTextArea textArea = new JTextArea(15, 50);
		textArea.setEditable(true);

		// Put it in a scroll pane in case the output is long.
		JScrollPane scrollPane = new JScrollPane(textArea);

		// The directory choosing button.
		JButton openButton = new JButton("Choose Directory");
		openButton.setVerticalTextPosition(AbstractButton.CENTER);
		openButton.setHorizontalTextPosition(AbstractButton.LEADING); // aka
																		// LEFT,
																		// for
																		// left-to-right
																		// locales
		openButton.setMnemonic(KeyEvent.VK_D);
		openButton.setActionCommand("disable");

		// The listener for openButton.
		ActionListener buttonListener = new FileChooserButtonListener(directoryFrame, directoryLabel, textArea,
				fileChooser);
		openButton.addActionListener(buttonListener);

		// Put it all together.
		Container c = directoryFrame.getContentPane();
		c.add(directoryLabel, BorderLayout.PAGE_START);
		c.add(scrollPane, BorderLayout.CENTER);
		c.add(openButton, BorderLayout.PAGE_END);

		directoryFrame.pack();
		return directoryFrame;
	}

	/**
	 * Create and show a directory explorer, which displays the contents of a
	 * directory.
	 *
	 * @param argsv
	 *            the command-line arguments.
	 */
	public static void main(String[] args) {
		DirectoryExplorer.buildWindow().setVisible(true);
	}

}
