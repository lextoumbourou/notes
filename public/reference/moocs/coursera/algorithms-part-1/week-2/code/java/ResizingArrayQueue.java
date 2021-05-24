public class ResizingArrayQueue {

	private String[] s;
	int head = 0;
	int tail = 0;

	public ResizingArrayQueue() {
		s = new String[1];
	}

	public void enqueue(String item) {
		s[tail] = item;

		tail = (tail + 1) % size();

		if (s[tail] != null) {
			resize();
		}
	}

	public String dequeue() {
		String output = s[head];
		s[head] = null;
		head = (head + 1) % size();

		return output;
	}

	public void resize() {
		String[] newS = new String[2 * s.length];

		for (int i = 0; i < s.length; i++) {
			newS[i] = s[i];
		}

		s = newS;
	}

	public int size() {
		return s.length;
	}
}
