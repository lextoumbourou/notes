# include <stdlib.h>
#include <stdio.h>

/*
 * Linked List from memory
 */

struct Node {
	int val;
	struct Node * next;
};

/*
 * Append item to start of Linked List.
 * 
 * Iterate through list until we get to last node.
 *
 * Malloc a node and append the pointer to it to the last node's next.
 */
void append(struct Node * head, int val) {
	struct Node * current_pos = head;

	while (current_pos->next != NULL) {
		current_pos = current_pos->next;
	}

	current_pos->next = malloc(sizeof(struct Node));

	current_pos->next->val = val;
	current_pos->next->next = NULL;
}

/*
 * Iterate through list and print each Node's val.
 */
void print_list(struct Node * head) {
	struct Node * current_pos = head;
	int counter = 1;

	printf("Printing list\n");

	while (current_pos->next != NULL) {
		printf("Val at pos %d: %d\n", counter, current_pos->val);
		current_pos = current_pos->next;
		counter += 1;
	}

	printf("Val at pos %d: %d\n", counter, current_pos->val);
}

/*
 * Push an item to the start of an linked list.
 *
 * Create a new node and assign the next position to
 * the head of the ll. Pointer the head of the ll to the new node.
 */
void push(struct Node ** head, int val) {
	struct Node * new_node = malloc(sizeof(struct Node));

	new_node->val = val;
	new_node->next = *head;

	*head = new_node;
}
