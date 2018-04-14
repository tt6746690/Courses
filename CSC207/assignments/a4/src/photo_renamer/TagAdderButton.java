package photo_renamer;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JList;

public class TagAdderButton extends ListSelectionButton<String> implements ActionListener {

	public TagAdderButton(String text, JList<String> source, PhotoRenamer p) {
		super(text, source, p);
	}
	
	public void actionPerformed(ActionEvent e){
		String selected = this.getSource().getSelectedValue();
		this.getP().getCurrentPhoto().addTagToPhoto(selected);
		this.getP().getPhotoTags().addElement(selected);
	}

}
