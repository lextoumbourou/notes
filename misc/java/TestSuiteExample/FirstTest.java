import static org.junit.Assert.*;
import org.junit.Test;

public class FirstTest {
	First f = new First();

	@Test
	public void testGetList() {
		String[] expected = {"Yadda", "Nadda", "Fadda"};
		String[] result = f.getList();
		assertArrayEquals(expected, result);
	}
}
