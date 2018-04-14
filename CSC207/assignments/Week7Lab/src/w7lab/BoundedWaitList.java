package w7lab;

public class BoundedWaitList<E> extends WaitList<E> {
	
	private int capacity;
	
	public BoundedWaitList(int capacity){
		super();
		this.capacity = capacity;
	}
	public int getCapacity(){
		return this.capacity;
	}
	@Override
	public void add(E element){
		if(this.content.size() < this.capacity){
			super.add(element);
		}
	}
	@Override
	public String toString(){
		return super.toString() + ". Capacity" + this.capacity;
	}

}
