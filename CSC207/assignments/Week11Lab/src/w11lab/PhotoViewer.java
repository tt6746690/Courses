package w11lab;

import javax.swing.*;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

/**
 * Additional reading/documentation:
 * https://docs.oracle.com/javase/tutorial/uiswing/layout/border.html
 * https://docs.oracle.com/javase/tutorial/uiswing/layout/visual.html
 * https://docs.oracle.com/javase/7/docs/api/javax/swing/JFileChooser.html
 *
 * For additional features:
 * http://docs.oracle.com/javase/tutorial/uiswing/events/documentlistener.html
 * http://www.java2s.com/Code/Java/2D-Graphics-GUI/ImageViewer.htm
 * https://docs.oracle.com/javase/tutorial/uiswing/components/list.html
 */
public class PhotoViewer {
	private final JFrame jframe;

	private final JPanel buttonContainer;
	private final JButton openFileButton;
	private final JButton renameFileButton;
	private final JTextArea prefixInput;

	private File selectedFile;
	private String prefix;

	private PhotoViewer() {
		this.jframe = new JFrame();
		openFileButton = new JButton("Choose directory");
		openFileButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				showFileChooser();
			}
		});
		renameFileButton = new JButton("Rename files");
		renameFileButton.addActionListener(new ActionListener() {
			@Override
			public void actionPerformed(ActionEvent e) {
				prefix = prefixInput.getText();
				System.out.println(prefix);
				renameFiles(selectedFile, prefix);
			}
		});

		renameFileButton.setEnabled(false);

		prefixInput = new JTextArea();

		Container content = this.jframe.getContentPane();

		// We create a new panel inside of our panel so that we can have
		// our buttons side by side, while also at the bottom of the main
		// layout.
		buttonContainer = new JPanel();
		buttonContainer.add(openFileButton, BorderLayout.LINE_START);
		buttonContainer.add(renameFileButton, BorderLayout.LINE_END);
		content.add(buttonContainer, BorderLayout.PAGE_END);
		content.add(prefixInput, BorderLayout.CENTER);
	}

	private void showFileChooser() {
		JFileChooser chooser = new JFileChooser();
		chooser.setCurrentDirectory(new File("."));
		chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
		int returnVal = chooser.showOpenDialog(jframe);
		if (returnVal == JFileChooser.APPROVE_OPTION) {
			selectedFile = chooser.getSelectedFile();
			renameFileButton.setEnabled(true);
		}
	}

	private void createAndShowGui() {
		// The following three lines will be included in basically every GUI
		// you see. If you don't include EXIT_ON_CLOSE, your application won't
		// close properly!
		jframe.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
		jframe.pack();
		jframe.setVisible(true);
	}

	private void renameFiles(File directory, String prefix) {
		for (File f : directory.listFiles()) {
			System.out.println("File in directory: " + f.getAbsolutePath());
			f.renameTo(new File(directory, prefix + "_" + f.getName()));
		}
	}

	public static void main(String[] args) {
		PhotoViewer v = new PhotoViewer();
		v.createAndShowGui();
	}
}