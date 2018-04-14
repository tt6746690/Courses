package photo_renamer;

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;

import javax.swing.Action;
import javax.swing.DefaultListModel;
import javax.swing.Icon;
import javax.swing.JButton;
import javax.swing.JTextField;

public class TextReaderButton extends JButton implements ActionListener{
	public JTextField source;
	public DefaultListModel<String> output;
	
	public TextReaderButton(String text, JTextField source, DefaultListModel<String> output){
		super(text);
		this.source = source;
		this.addActionListener(this);
		this.output = output;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		String sourcetext = this.source.getText();
		try {
			if(PhotoNode.createNewTag(sourcetext)){
				output.addElement(sourcetext);
			}
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}		
	}	
}
