class TestQuickFind {
	public static void main(String[] args) {
		QuickFind qf = new QuickFind(8);
		qf.union(0, 1);
		if (!qf.connected(0, 1)) {
			System.out.println("Test failed: 0 and 1 not connected");
			return;
		}

		qf.union(1, 7);
		if (!qf.connected(0, 7)) {
			System.out.println("Test failed: 0 and 7 not connected");
			return;
		}

		System.out.println("All tests passed.");
	}
}
