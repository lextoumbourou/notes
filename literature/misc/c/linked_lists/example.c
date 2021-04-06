# include <stdlib.h>
# include <stdio.h>
# include <assert.h>

/*
 * A Linked List implementation based on reading here: http://www.learn-c.org/en/Linked_lists
 */

struct Node {
	int val;
	struct Node * next;
};

typedef struct Node Node_t;

/*
 * Append an item to the start of a linked list.
 *
 * @param head - a pointer to the start of the LL.
 * @param val - an integer value to set at end of list.
 */
void append(Node_t * head, int val) {
	Node_t * current = head;

	// Iterate through loop until current is last node.
	while (current->next != NULL) {
		current = current->next;
	}

	// Allocate some memory for the end and set it.
	current->next = malloc(sizeof(Node_t));
	current->next->val = val;
	current->next->next = NULL;
}

/*
 * Push an item to start of list.
 *
 * @param head - a pointer to the pointer of the start of the LL.
 * @param val - an integer value to set in the new node.
 */
void push(Node_t ** head, int val) {
	// Allocate some memory for the new node
	Node_t * new_node = malloc(sizeof(Node_t));

	new_node->val = val;

	// Set new node's next to the previous head's pointer.
	new_node->next = *head;

	// Set the head pointer to a pointer to the new node.
	*head = new_node;
}

/*
 * Take an item from the front of list and return the value stored in it.
 *
 * @param head - a pointer to the pointer at the start of the LL.
 */
int pop(Node_t ** head) {
	int retval = (*head)->val;

	Node_t * next_node = NULL;

	if (*head == NULL) {
		return -1;
	}

	// If the current head is the end, then delete it.
	if ((*head)->next == NULL) {

		next_node = *head;
		
	} else {
		// Otherwise set it to the next item in the list.
		next_node = (*head)->next;
	}

	free(*head);

	*head = next_node;

	return retval;
}

/*
 * Remove the last item from a linked list and return the value.
 *
 * @param head - a pointer to the start of the LL.
 */
int remove_last(Node_t * head) {
	int retval = -1;

	// If we're at the end, then delete the head and return.
	if (head->next == NULL) {
		retval = head->val;
		free(head);
		head = NULL;
		return retval;
	}


	Node_t * current = head;

	// Loop until we get to one before the last one.
	while (current->next->next != NULL) {
		current = current->next;
	}

	retval = current->next->val;

	// 2nd last item becomes last and we free the last.
	free(current->next);
	current->next = NULL;

	return retval;
}

/*
 * Print every val in a linked list.
 *
 * @param head - a pointer to the start of the LL.
 */
void print_list(Node_t * head) {
	Node_t * current = head;

	while (current != NULL) {
		printf("%d\n", current->val);
		current = current->next;
	}
}

int main() {
	Node_t * head;
	int last;
	int popped;

	head = malloc(sizeof(Node_t));

	head->val = 2;
	head->next = NULL;

	push(&head, 1);
	append(head, 3);
	append(head, 4);

	popped = pop(&head);
	assert(popped == 1);

	last = remove_last(head);
	assert(last == 4);

	last = remove_last(head);
	assert(last == 3);

	return 0;
}
