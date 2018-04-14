package photo_renamer;

import static org.junit.Assert.*;

import java.io.File;
import java.io.IOException;

import org.junit.BeforeClass;
import org.junit.Test;

public class PhotoNodeTest {
	
	/*
	protected void setUp(){
		try {
			PhotoNode.initializeTags();
			System.out.println("the f*ck?");
		} catch (IOException e) {
			e.printStackTrace();
		} 
		//File f = new File ("/Users/anonymous/Desktop/photos/@Meep @Toronto @Boom sign-check-icon.png");
		//PhotoNode p = new PhotoNode("sign-check-icon.png", null, f);
	}
	*/

	@Test
	public void testInitializeTags() {
		try {
			PhotoNode.initializeTags();
		} catch (IOException e) {
			e.printStackTrace();
		} 
		for(String t: PhotoNode.allTags){
			System.out.println(t);
		}
		assertTrue(PhotoNode.allTags.contains("@Toronto"));
	}
	
	
	@Test
	public void testCreateNewTags1() {
		int originalSize;
		if (!PhotoNode.allTags.contains("@Kingston")){
			originalSize = PhotoNode.allTags.size();
		}
		else{
			originalSize = PhotoNode.allTags.size() - 1;
		}
		try {
			PhotoNode.createNewTag("@Kingston");
		} catch (IOException e) {
			e.printStackTrace();
		}
		assertTrue(PhotoNode.allTags.contains("@Kingston") && PhotoNode.allTags.size() == originalSize + 1);
	}
	
	@Test
	public void testCreateNewTags2() {
		int originalSize = PhotoNode.allTags.size();
		try {
			PhotoNode.createNewTag("Waterloo");
		} catch (IOException e) {
			e.printStackTrace();
		}
		//Look at me! I passed CSC165, and I know what DeMorgan's Laws are!
		//(Also, my partner never passed CSC165).
		//(Which is TECHNICALLY true, but only because she was too smart for it, and skipped ahead
		// and took CSC240).
		assertTrue(!((PhotoNode.allTags.contains("Waterloo")) || (PhotoNode.allTags.size() == originalSize + 1)));
	}
	
	@Test
	public void testCreateNewTags3() {
		int originalSize = PhotoNode.allTags.size();
		try {
			PhotoNode.createNewTag("@St Johns");
		} catch (IOException e) {
			e.printStackTrace();
		}
		assertTrue(!((PhotoNode.allTags.contains("@St Johns")) || (PhotoNode.allTags.size() == originalSize + 1)));
	}
	
	@Test
	public void testCreateNewTags4() {
		int originalSize = PhotoNode.allTags.size();
		try {
			PhotoNode.createNewTag("@@@");
		} catch (IOException e) {
			e.printStackTrace();
		}
		assertTrue(!((PhotoNode.allTags.contains("@@@")) || (PhotoNode.allTags.size() == originalSize + 1)));
	}
	
	@Test
	public void testCreateNewTags5() {
		int originalSize = PhotoNode.allTags.size();
		try {
			PhotoNode.createNewTag("@Montreal");
		} catch (IOException e) {
			e.printStackTrace();
		}
		assertTrue(PhotoNode.allTags.size() == originalSize);
	} 
	
	/*
	@Test
	public void testdeleteTag1() {
		int originalSize = PhotoNode.allTags.size();
		try {
			PhotoNode.createNewTag("@Calgary");
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		try {
			PhotoNode.deleteTag("@Calgary");
		} catch (IOException e) {
			e.printStackTrace();
		}
		assertTrue(!(PhotoNode.allTags.contains("@Calgary")) && (PhotoNode.allTags.size() == originalSize - 1));
	} */

}
