package photo_renamer;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.AbstractButton;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;

public class DirectoryChooserButton extends JButton implements ActionListener {

	private PhotoRenamer p;
	private JFrame dFrame;
	
	/**
	 * Initialize a DirectoryChooserButton
	 * 
	 * @param text
	 * 			The name of the button.
	 * @param p
	 * 			The PhotoRenamer that this DirectoryChooserButton should appear in.
	 */
	public DirectoryChooserButton(String text, PhotoRenamer p) {
		super(text);
		this.p = p;
		this.addActionListener(this);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		//this.dFrame = this.buildWindow();
		//this.dFrame.setVisible(true);
		
	}

}
