#include <stdio.h>

int
main(void)
{
    int x = 5;
    int *y = &x;
    printf("x = %d\n", x);
    printf("y = %p\n", y);

    printf("x is stored at %p\n", &x);
    printf("x is stored at %p\n", y);
}
