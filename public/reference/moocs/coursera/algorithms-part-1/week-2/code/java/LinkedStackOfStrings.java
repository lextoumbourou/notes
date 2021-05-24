public class LinkedStackOfStrings {

	private Node first = null;

	private class Node {
        	String item;
        	Node next;
	}

	public boolean isEmpty() {
		return first == null;
	}

	/*
	 * Create a temporary reference to first node,
	 * create a new first node and make ``next`` a reference
	 * to temporary first node.
	 */
	public void push(String item) {
		Node oldfirst = first;
		first = new Node();
		String first.item = item;
		first.next = oldfirst;
	}

	/*
	 * Take the item from the first node
	 * and make first a reference to the next node,
	 * then return the item.
	 */
	public String pop() {
		String item = first.item;
		first = first.next;
		return item;
	}
}
