public class ResizingArrayStackOfStrings {
	private String[] s;
	private int N = 0;

	public ResizingArrayStackOfStrings() {
		s = new String[1];
	}

	public void push(String item)
	{
		if (N == s.length) resize(2 * s.length);

		s[N++] = item;
	}

	public String pop()
	{
		String output = s[--N];
		s[N] = null;
		if (N > 0 && N == s.length / 4) resize(s.length / 2);
		return output;
	}

	public int size()
	{
		return s.length;
	}

	private void resize(int capacity)
	{
		String[] copy = new String[capacity];
		for (int i = 0; i < N; i++)
			copy[i] = s[i];

		s = copy;
	}
}
