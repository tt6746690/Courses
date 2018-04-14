package photo_renamer;

import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTextField;

public class RevertToButton extends JButton implements ActionListener{
	public PhotoRenamer p;
	public JFrame logFrame;
	
	public RevertToButton(String text, PhotoRenamer p){
		super(text);
		this.addActionListener(this);
		this.p = p;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		this.buildLogFrame();
		this.revertOptions();
		this.logFrame.setVisible(true);
	}
	
	private void buildLogFrame(){
		this.logFrame = new JFrame("Log");
		JPanel logPanel = new JPanel();
		JLabel logLabel = new JLabel();
		logPanel.setPreferredSize(new Dimension(1000, 500));
		String name = p.getCurrentPhoto().actual_file.getName();
		LogEntry l = PhotoRenamerShell.getLog(name);
		String log = l.toString();
		logLabel.setText(log);
		logPanel.add(logLabel, BorderLayout.CENTER);
	
		this.logFrame.add(logPanel, BorderLayout.CENTER);
		this.logFrame.pack();
	}
	
	private void revertOptions(){
		JPanel indexPanel = new JPanel();
		indexPanel.setPreferredSize(new Dimension(400, 100));
		indexPanel.setMaximumSize(new Dimension(400, 100));
		indexPanel.setMinimumSize(new Dimension(400, 100));
		indexPanel.setLayout(new BoxLayout(indexPanel, BoxLayout.PAGE_AXIS));
		JTextField indexField = new JTextField();
		indexField.setMaximumSize(new Dimension(100, 40));
		indexField.setMinimumSize(new Dimension(100, 40));
		indexField.setMaximumSize(new Dimension(100, 40));
		indexPanel.add(indexField);
		RevertToIndexButton rib = new RevertToIndexButton("Confirm Revert", this.p, indexField, this);
		indexPanel.add(rib);
		this.logFrame.add(indexPanel, BorderLayout.SOUTH);
	}
	
}


