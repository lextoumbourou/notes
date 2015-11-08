#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
	int total = 0;
	char *tmp;

	if (argc == 1) {
		printf("Usage:\n   ./brute_force_3_sum integers...\n");
		return 1;
	}
	if (argc < 4) {
		printf("ERROR: Need at least 3 integers to run this.\n");
		return 2;
	}

	for (int i = 1; i < argc; i++) {
		// Convert i to integer (no error checking).
		long int_i = strtol(argv[i], &tmp, 10);

		for (int j = i + 1; j < argc; j++) {
			// Convert j to integer (no error checking).
			long int_j = strtol(argv[j], &tmp, 10);

			for (int k = j + 1; k < argc; k++) {
				// Convert k to integer (no error checking).
				long int_k = strtol(argv[k], &tmp, 10);

				long sum = (int_i + int_j + int_k);
				if (sum == 0) {
					printf(
						"DEBUG: Found combination %ld + %ld + %ld that returns 0\n",
						int_i, int_j, int_k);
					total++;
				}
			}
		}
	}

	printf("The total number of 3-sum triplets: %d\n", total);

	return 0;
}
