# include <stdio.h>

int main() {
	int i = 2;
	int j = 3;
	printf("Init: %d\n", i);

	i = j + --i;
	printf("Take J from --i: %d\n", i);

	i = 2;
	j = 3;
	i = j + i--;

	printf("Take one from i--: %d\n", i);
	return 0;
}
