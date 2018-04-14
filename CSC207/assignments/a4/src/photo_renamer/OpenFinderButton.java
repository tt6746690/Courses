package photo_renamer;

import java.awt.Desktop;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;
import java.io.IOException;

import javax.swing.JButton;

public class OpenFinderButton extends JButton implements ActionListener{

	PhotoRenamer p;
	//TODO write constructor
	public OpenFinderButton(String text, PhotoRenamer p){
		super(text);
		this.addActionListener(this);
		this.p = p;
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		//TODO
		File path = new File(p.getCurrentPhoto().openInFinder());
		try {
			Desktop.getDesktop().open(path);
			System.out.println(path);
		} catch (IOException e1) {
			System.out.println("IOException in actionPerformed in OpenFinderButton!");
		}
	}

}
