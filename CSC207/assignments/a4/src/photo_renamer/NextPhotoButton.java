package photo_renamer;

import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.DefaultListModel;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;

public class NextPhotoButton extends JButton implements ActionListener{

	private PhotoRenamer p;
	
	public NextPhotoButton(String text, PhotoRenamer p){
		super(text);
		this.p = p;
		this.addActionListener(this);
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		//Cycle to the next image
		this.p.nextPhoto();
		
		//Update the name displayed
		this.p.noTagName.setText("Name without tags: " + 
				PhotoNode.stripOfTags(this.p.getCurrentPhoto().actual_file.getName()) + "\n");
		this.p.yesTagName.setText("Name with tags: " + this.p.getCurrentPhoto().actual_file.getName());
		
		//Update the actual image
		this.p.updateIcon();
		this.p.imageLabel.setIcon(this.p.imIcon);
		this.p.actualImage.removeAll();
		this.p.actualImage.add(this.p.imageLabel);
		
		//Update the list of tags displayed at the west side of the GUI.
		this.p.getPhotoTags().removeAllElements();
		for (String tag : this.p.getCurrentPhoto().getTags()) {
			this.p.getPhotoTags().addElement(tag);
		}
	}

}
