package photo_renamer;

import java.util.ArrayList;

public class Tag {
	private ArrayList<String> tags;

	public void newTag(String newTag){
		tags.add(newTag);
	}
	
	public void deleteTag(String toDelete){
		tags.remove(toDelete);
	}
	
	public static void main(String[] args){
		System.out.println("Meep");
	}
	
}
