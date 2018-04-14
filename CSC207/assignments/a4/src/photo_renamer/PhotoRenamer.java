
package photo_renamer;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Image;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

import javax.imageio.ImageIO;
import javax.swing.AbstractButton;
import javax.swing.BoxLayout;
import javax.swing.DefaultListModel;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTextArea;
import javax.swing.JTextField;

/**
 * Create and show a directory explorer, which displays the contents of a
 * directory.
 * 
 * Design pattern: PhotoRenamer is a Singleton class; there should be exactly one instance of this class at one time.
 */
public class PhotoRenamer {

	private ArrayList<PhotoNode> photos; // Design pattern: Composite: PhotoRenamer "has" a list of PhotoNodes
	/** Design patter: Singleton
	 * This is the universal copy of the PhotoRenamer object.
	 */
	private static final PhotoRenamer instance = new PhotoRenamer();
	private PhotoNode currentPhoto;
	private int currentIndex;
	private File directory;
	private JList<String> masterlist;
	private JList<String> currentTags; // The tags belonging to the current photo
	private DefaultListModel<String> photoTags;
	private DefaultListModel<String> jMasterList;
	private JFrame dWindow;
	private JFrame iFrame = new JFrame();
	private JPanel panelEast = new JPanel();
	private JPanel panelCenter = new JPanel();
	private JPanel panelWest = new JPanel();

	/** Attributes pertinent to the name of the file currently being viewed.
	*/
	protected JPanel imageInformation = new JPanel();
	protected JLabel noTagName = new JLabel();
	protected JLabel yesTagName = new JLabel();

	// Attributes pertinent to the display of the image.
	protected JPanel actualImage = new JPanel();
	protected JLabel imageLabel = new JLabel();
	protected ImageIcon imIcon = new ImageIcon();
	
	//Attributes pertinent to the display of the current image's tags.
	protected JScrollPane currentTagsScrollPane = new JScrollPane();

	/**
	 * Constructor; private since this since this is a Singleton class
	 */
	private PhotoRenamer() {
		this.currentIndex = 0;
	}

	/** Constructor that takes in a parameter file; mostly for testing purposes;
	 * private since this since this is a Singleton class
	 * 
	 * @param file
	 * @throws UnexpectedFileTypeException
	 * @throws IOException
	 */
	private PhotoRenamer(File file) throws UnexpectedFileTypeException, IOException {
		this.currentIndex = 0;
		this.setDirectory(file);
		// Generate the list of files we need to display.
		photos = PhotoRenamerShell.readySet(this.getDirectory());
		this.setCurrentPhoto(photos.get(currentIndex));
	}
	
	/** Gives the user the universal copy of an instance of this Singleton class
	 * 
	 * @return PhotoRenamer
	 */
	public static PhotoRenamer getInstanceOf(){
		return instance;
	}

	
	public boolean usingTag(String tag){
		boolean usingTag = false;
		for(PhotoNode p: photos){
			if(p.getTags().contains(tag)){
				usingTag = true;
			}
		}
		return usingTag;
	}
	
	/**
	 * Create and return the window for the directory explorer.
	 *
	 * @return JFrame; the window for the directory explorer
	 */

	public JFrame buildDWindow() {
		JFrame directoryFrame = new JFrame("Directory Chooser");
		directoryFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);

		JLabel directoryLabel = new JLabel("Select a directory to view and rename photos");

		// Set up the area for the directory contents.
		JTextArea textArea = new JTextArea(15, 50);
		textArea.setEditable(true);

		// Put it in a scroll pane in case the output is long.
		JScrollPane scrollPane = new JScrollPane(textArea);

		// The directory choosing button.
		JButton openButton = new JButton("Choose Directory");
		openButton.setVerticalTextPosition(AbstractButton.CENTER);
		openButton.setHorizontalTextPosition(AbstractButton.LEADING);
		// openButton.setMnemonic(KeyEvent.VK_D);
		openButton.setActionCommand("disable");

		// The listener for openButton.
		ActionListener buttonListener = new FileChooserButtonListener(directoryFrame, directoryLabel, fileChooser,
				this);
		openButton.addActionListener(buttonListener);

		// Put it all together.
		Container c = directoryFrame.getContentPane();
		c.add(directoryLabel, BorderLayout.PAGE_START);
		c.add(scrollPane, BorderLayout.CENTER);
		c.add(openButton, BorderLayout.PAGE_END);

		directoryFrame.pack();
		return directoryFrame;
	}

	/** Builds the main window of our program, in which the user views and edits the images
	 * 
	 * @throws UnexpectedFileTypeException
	 * @throws IOException
	 */
	public void buildImageTagger() throws UnexpectedFileTypeException, IOException {
		if (this.getDirectory().isDirectory()) {

			// Generate the list of files we need to display.
			this.photos = PhotoRenamerShell.readySet(this.getDirectory());
			this.setCurrentPhoto(photos.get(currentIndex));

			// Also, PhotoNode's alltags variable should now be updated.

			// JFrame iFrame = new JFrame("Image Tagger");
			this.setiFrame(new JFrame("Image Tagger"));
			getiFrame().setSize(2000, 800);
			getiFrame().setLocationRelativeTo(null);
			getiFrame().setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

			this.setPanelWest(new JPanel());

			this.getPanelEast().setBackground(Color.ORANGE);
			// panelEast.setBounds(1150, 0, 300, 1000);

			this.getPanelWest().setBackground(Color.ORANGE);
			this.getPanelWest().setPreferredSize(new Dimension(400, 100));
			this.getPanelCenter().setBackground(Color.BLACK);
			this.getPanelEast().setPreferredSize(new Dimension(400, 100));

			getiFrame().add(this.getPanelWest(), BorderLayout.LINE_START);
			getiFrame().add(this.getPanelEast(), BorderLayout.LINE_END);
			getiFrame().add(this.getPanelCenter(), BorderLayout.CENTER);

			// Setting layout for all the panels
			this.getPanelEast().setLayout(new BoxLayout(this.getPanelEast(), BoxLayout.Y_AXIS));
			this.getPanelCenter().setLayout(new BoxLayout(this.getPanelCenter(), BoxLayout.Y_AXIS));
			this.getPanelWest().setLayout(new BoxLayout(this.getPanelWest(), BoxLayout.Y_AXIS));

			this.buildPanelEast();

			// Adding a "current image's tags" label
			JLabel currentTags = new JLabel("Current Image's Tags");
			this.getPanelWest().add(currentTags);

			// Making content for panelWest
			this.buildPhotoTagsPanel();
			this.getPanelWest().add(this.currentTagsScrollPane);
			this.getPanelWest().add(buildTagFunctionsPanel());

			// Displaying the actual image and adding the image info
			this.buildImagePanel();
			this.getPanelCenter().add(this.actualImage);
			this.buildImageInfoPanel();
			this.getPanelCenter().add(this.imageInformation);

			// return iFrame;
		} else {
			// return null;
		}
	}

	/** Builds one of the panels in iFrame
	 * 
	 */
	public void buildPanelEast() {
		// Adding a "master list of tags" label
		JLabel masterList = new JLabel("Master List of Tags");
		this.getPanelEast().add(masterList);

		// The master list of tags + adding and removing
		setjMasterList(new DefaultListModel<String>());
		// Loop through allTags and add everything to the jMasterList
		for (String tag : PhotoNode.allTags) {
			getjMasterList().addElement(tag);
		}
		masterlist = new JList<String>(getjMasterList());
		JScrollPane listScroller = new JScrollPane(masterlist);
		listScroller.setPreferredSize(new Dimension(100, 100));
		this.getPanelEast().add(listScroller);

		JTextField newTagField = new JTextField();
		newTagField.setMaximumSize(new Dimension(200, 60));
		this.getPanelEast().add(newTagField);

		JPanel buttonPanel1 = new JPanel();
		buttonPanel1.setPreferredSize(new Dimension(20, 100));
		buttonPanel1.setBackground(Color.ORANGE);
		TextReaderButton newTagButton = new TextReaderButton("New Tag", newTagField, getjMasterList());
		buttonPanel1.add(newTagButton);
		TagDeleterButton deleteTagButton = new TagDeleterButton("Delete Tag", this.masterlist, this);
		buttonPanel1.add(deleteTagButton);
		this.getPanelEast().add(buttonPanel1);
	}

	public void updateIcon() {
		// Create the file representing the desired image
		File source = new File(this.currentPhoto.openInFinder() + "/" + this.currentPhoto.getNom());
		System.out.println(source);
		// Create the image file that represents the file
		BufferedImage img = null;
		try {
			img = ImageIO.read(source);
		} catch (IOException e) {
			System.out.print("IOException in buildImagePanel");
		}
		// Create the image icon that represents the image.
		ImageIcon icon = new ImageIcon(img);
		this.imageLabel = new JLabel(null, icon, JLabel.CENTER);
		Image img1 = img.getScaledInstance(400, 400, Image.SCALE_SMOOTH);
		ImageIcon icon1 = new ImageIcon(img1);
		this.imIcon = icon1;
	}

	public void buildImagePanel() {
		JPanel imagePanel = new JPanel();
		imagePanel.setPreferredSize(new Dimension(400, 400));
		this.updateIcon();
		this.imageLabel = new JLabel(null, this.imIcon, JLabel.CENTER);
		imagePanel.add(imageLabel);
		this.actualImage = imagePanel;
	}

	private void buildImageInfoPanel() {
		JPanel imageInfoPanel = new JPanel();
		imageInfoPanel.setPreferredSize(new Dimension(200, 300));
		imageInfoPanel.setBackground(Color.cyan);
		imageInfoPanel.setLayout(new BoxLayout(imageInfoPanel, BoxLayout.Y_AXIS));

		this.noTagName
				.setText("Name without tags: " + PhotoNode.stripOfTags(this.currentPhoto.actual_file.getName()) + "\n");
		this.yesTagName.setText("Name with tags: " + this.currentPhoto.actual_file.getName());
		imageInfoPanel.add(this.noTagName);
		imageInfoPanel.add(this.yesTagName);

		// Making buttons
		OpenFinderButton finderButton = new OpenFinderButton("Open in Finder", this);
		imageInfoPanel.add(finderButton);
		NextPhotoButton nextPhoto = new NextPhotoButton("Next Photo", this);
		imageInfoPanel.add(nextPhoto);
		PreviousPhotoButton previousPhoto = new PreviousPhotoButton("Previous Photo", this);
		imageInfoPanel.add(previousPhoto);
		SeeLogButton seeLog = new SeeLogButton("See Log", this);
		imageInfoPanel.add(seeLog);
		;
		this.imageInformation = imageInfoPanel;
	}

	public void buildPhotoTagsPanel() {
		this.setPhotoTags(new DefaultListModel<String>());
		for (String tag : this.currentPhoto.getTags()) {
			this.getPhotoTags().addElement(tag);
			System.out.println(tag);
		}
		this.setCurrentTags(new JList<String>(this.getPhotoTags()));
		JScrollPane listScroller = new JScrollPane(this.getCurrentTags());
		listScroller.setPreferredSize(new Dimension(100, 100));
		listScroller.setMaximumSize(new Dimension(400, 400));
		listScroller.setMinimumSize(new Dimension(400, 400));
		this.currentTagsScrollPane = listScroller;
	}

	public JPanel buildTagFunctionsPanel() {
		JPanel tf = new JPanel();
		tf.setLayout(new BoxLayout(tf, BoxLayout.Y_AXIS));
		TagAdderButton tagAdder = new TagAdderButton("Add tag", this.masterlist, this);
		tf.add(tagAdder);
		TagRemoverButton tagDeleter = new TagRemoverButton("Remove tag", this.getCurrentTags(), this);
		tf.add(tagDeleter);
		ConfirmRenameButton confirmRename = new ConfirmRenameButton("Confirm Rename", this);
		tf.add(confirmRename);
		RevertToButton revertTo = new RevertToButton("Revert To", this);
		tf.add(revertTo);
		tf.setPreferredSize(new Dimension(100, 100));
		tf.setMaximumSize(new Dimension(200, 400));
		tf.setMinimumSize(new Dimension(200, 400));
		return tf;
	}

	public PhotoNode getCurrentPhoto() {
		return currentPhoto;
	}

	public void setCurrentPhoto(PhotoNode currentPhoto) {
		this.currentPhoto = currentPhoto;
	}
	
	public File getDirectory() {
		return directory;
	}

	public void setDirectory(File directory) {
		this.directory = directory;
	}

	public void nextPhoto() {
		if (this.currentIndex < this.photos.size()-1) {
			this.currentIndex += 1;
			this.currentPhoto = this.photos.get(this.currentIndex);
		} else {
			System.out.println("No more photos to view in this directory!");
		}
	}

	public void previousPhoto() {
		if (this.currentIndex > 0) {
			this.currentIndex -= 1;
			this.currentPhoto = this.photos.get(this.currentIndex);
		}else{
			System.out.println("No previous photo!");
		}
	}

	JList<String> getCurrentTags() {
		return currentTags;
	}

	void setCurrentTags(JList<String> currentTags) {
		this.currentTags = currentTags;
	}

	DefaultListModel<String> getPhotoTags() {
		return photoTags;
	}

	void setPhotoTags(DefaultListModel<String> photoTags) {
		this.photoTags = photoTags;
	}

	DefaultListModel<String> getjMasterList() {
		return jMasterList;
	}

	void setjMasterList(DefaultListModel<String> jMasterList) {
		this.jMasterList = jMasterList;
	}

	JFrame getdWindow() {
		return dWindow;
	}

	void setdWindow(JFrame dWindow) {
		this.dWindow = dWindow;
	}

	JFrame getiFrame() {
		return iFrame;
	}

	void setiFrame(JFrame iFrame) {
		this.iFrame = iFrame;
	}

	JPanel getPanelEast() {
		return panelEast;
	}

	void setPanelEast(JPanel panelEast) {
		this.panelEast = panelEast;
	}

	JPanel getPanelCenter() {
		return panelCenter;
	}

	void setPanelCenter(JPanel panelCenter) {
		this.panelCenter = panelCenter;
	}

	JPanel getPanelWest() {
		return panelWest;
	}

	void setPanelWest(JPanel panelWest) {
		this.panelWest = panelWest;
	}
	
	/** Main method
	 * 
	 * @param args
	 * @throws IOException
	 * @throws UnexpectedFileTypeException
	 */
	public static void main(String[] args) throws IOException, UnexpectedFileTypeException {
		//File f = new File("/Users/anonymous/Desktop/photos");
		//PhotoRenamer w = new PhotoRenamer(f);
		// System.out.println(w.currentPhoto.getNom());
		//w.buildImageTagger();
		//w.iFrame.setVisible(true);
		PhotoRenamer w = PhotoRenamer.getInstanceOf();
		w.setdWindow(w.buildDWindow());
		w.getdWindow().setVisible(true);
	}
	
}
