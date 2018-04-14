package photo_renamer;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JList;

public class TagRemoverButton extends ListSelectionButton<String> implements ActionListener {

	public TagRemoverButton(String text, JList<String> source, PhotoRenamer p) {
		super(text, source, p);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		String selected = this.getSource().getSelectedValue();
		this.getP().getCurrentPhoto().deleteTagFromPhoto(selected);
		this.getP().getPhotoTags().removeElement(selected);
	}

}
