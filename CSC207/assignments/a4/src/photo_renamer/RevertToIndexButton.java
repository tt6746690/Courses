package photo_renamer;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JTextField;

public class RevertToIndexButton extends JButton implements ActionListener {

	private PhotoRenamer p;
	private JTextField source;
	private RevertToButton r;
	
	public RevertToIndexButton(String text, PhotoRenamer p, JTextField source, RevertToButton r){
		super(text);
		this.p = p;
		this.addActionListener(this);
		this.source = source;
		this.r = r;
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		String input = this.source.getText();
		int index = Integer.parseInt(input);
		LogEntry m = PhotoRenamerShell.getLog(this.p.getCurrentPhoto().actual_file.getName());
		String revertedName = m.getRevertedName(index);
		System.out.println(revertedName);
		this.p.getCurrentPhoto().confirmRename(revertedName);
		m.newName(revertedName);
		PhotoRenamerShell.serializeLog();
		this.p.yesTagName.setText(revertedName);
		this.r.logFrame.setVisible(false);
	}

}
