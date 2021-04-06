#include <assert.h>
#include "example_2.c"


int main() {
	struct Node * test_node = malloc(sizeof(struct Node));
	test_node->val = 4;
	test_node->next = NULL;

	assert(test_node->val == 4);

	append(test_node, 5);

	assert(test_node->next->val == 5);

	push(&test_node, 10);

	assert(test_node->val == 10);

	return 0;
}
