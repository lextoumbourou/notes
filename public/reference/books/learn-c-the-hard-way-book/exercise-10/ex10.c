#include <stdio.h>

int main(int argc, char *argv[])
{
    int i = 0;

    for (i = 1; i < argc; i++) {
        printf("argd %d: %s\n", i, argv[i]);
    }

    char *states[] = {
        "Victoria", "NSW",
        "South Australia"
    };

    int num_states = 3;

    for (i = 0; i < num_states; i++) {
        printf("state %d: %s\n", i, states[i]);
    };

    return 0;
}

