package photo_renamer;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

public class SeeLogButton extends JButton implements ActionListener{

	private PhotoRenamer p;
	
	public SeeLogButton(String text, PhotoRenamer p){
		super(text);
		this.p = p;
		this.addActionListener(this);
	}
	
	@Override
	public void actionPerformed(ActionEvent e) {
		this.buildLogFrame().setVisible(true);
	}

	public JFrame buildLogFrame(){
		JFrame logFrame = new JFrame("Log");
		JPanel logPanel = new JPanel();
		JLabel logLabel = new JLabel();
		logPanel.setPreferredSize(new Dimension(1000, 600));
		String name = p.getCurrentPhoto().actual_file.getName();
		LogEntry l = PhotoRenamerShell.getLog(name);
		String log = l.toString();
		logLabel.setText(log);
		logPanel.add(logLabel, BorderLayout.CENTER);
		logFrame.add(logPanel);
		logFrame.pack();
		return logFrame;
	}
}
