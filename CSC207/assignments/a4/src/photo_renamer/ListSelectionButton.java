package photo_renamer;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.JButton;
import javax.swing.JList;

public abstract class ListSelectionButton<T> extends JButton implements ActionListener{
	private JList<T> source;
	private PhotoRenamer p;
	
	/**
	 * 
	 * 
	 * @param text
	 * @param source
	 * @param p
	 */
	public ListSelectionButton(String text, JList<T> source, PhotoRenamer p) {
		super(text);
		this.setSource(source);
		this.setP(p);
		this.addActionListener(this);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		T selected = this.getSource().getSelectedValue();
	}

	public JList<T> getSource() {
		return source;
	}

	public void setSource(JList<T> source) {
		this.source = source;
	}

	public PhotoRenamer getP() {
		return p;
	}

	public void setP(PhotoRenamer p) {
		this.p = p;
	}

}
