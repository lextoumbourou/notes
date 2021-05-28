public class TestResizingArrayQueue {
	public static void main(String[] args) {
		ResizingArrayQueue ra = new ResizingArrayQueue();
		System.out.println("Array size should be 1. Actual size: " + ra.size());

		ra.enqueue("hello");
		System.out.println("Doubled array size should be 2. Actual size: " + ra.size());

		ra.enqueue("world");
		ra.enqueue("sup?");
		System.out.println("Doubled array size should be 4. Actual size: " + ra.size());

		String item = ra.dequeue();
		System.out.println("This is the item: " + item);
	}
}
