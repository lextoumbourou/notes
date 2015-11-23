public class TestResizingArrayStackOfStrings {
	public static void main(String[] args) {
		ResizingArrayStackOfStrings ra = new ResizingArrayStackOfStrings();

		ra.push("Hello");
		ra.push("World");

		if (ra.size() != 2) {
			System.out.println("Test failed. Expected size: 2, actual size: " + ra.size());
			return;
		}

		ra.push("Sup breh?");

		// Should double at this point!
		if (ra.size() != 4) {
			System.out.println("Test failed. Expected size: 4, actual size: " + ra.size());
			return;
		}


		String item = ra.pop();
		if (item != "Sup breh?") {
			System.out.println("Test failed. Expected item: 'World', actual item: " + item);
			return;
		}
	}
}
