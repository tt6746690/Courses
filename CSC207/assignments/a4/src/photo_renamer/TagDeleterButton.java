package photo_renamer;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;

import javax.swing.JList;

public class TagDeleterButton extends ListSelectionButton<String> implements ActionListener {

	public TagDeleterButton(String text, JList<String> source, PhotoRenamer p) {
		super(text, source, p);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		String selected = this.getSource().getSelectedValue();
		try {
			if (PhotoNode.deleteTag(selected)) {
				this.getP().getjMasterList().removeElement(selected);
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		}

	}

}
