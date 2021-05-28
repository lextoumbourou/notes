#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

pthread_t tid[5];

void* doSomeThing(void *arg)
{
	while (1) {
		printf("I'm doing something in a thread.\n");
	}
}

int main(void)
{
	int i = 0;
	int err;

	while(i < 5)
	{
		err = pthread_create(&(tid[i]), NULL, &doSomeThing, NULL);
		i++;
	}

	sleep(100);

	return 0;
}
