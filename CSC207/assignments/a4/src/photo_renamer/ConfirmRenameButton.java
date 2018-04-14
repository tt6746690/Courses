package photo_renamer;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.Action;
import javax.swing.Icon;
import javax.swing.JButton;

public class ConfirmRenameButton extends JButton implements ActionListener {
	
	private PhotoRenamer p;

	/** If clicked, this button confirms the renaming of the image to 
	 * include all of its tags and logs the changes in name.
	 * 
	 * @param text
	 * @param p
	 */
	public ConfirmRenameButton(String text, PhotoRenamer p) {
		super(text);
		this.p = p;
		this.addActionListener(this);
	}

	/** Confirms the renaming of the image to 
	 * include all of its tags and logs the changes in name.
	 * 
	 */
	@Override
	public void actionPerformed(ActionEvent e) {
		String oldName = this.p.getCurrentPhoto().actual_file.getName();
		String newName = this.p.getCurrentPhoto().confirmRename();
		LogEntry l = PhotoRenamerShell.getLog(oldName);
		l.newName(newName);
		this.p.yesTagName.setText(newName);
		PhotoRenamerShell.serializeLog();
	}

}
