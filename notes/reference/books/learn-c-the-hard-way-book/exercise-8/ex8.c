# include <stdio.h>

int main(int argc, char *argv[])
{
    int areas[] = {10, 100, 13, 23, 123};
    char name[] = "Travis";
    char full_name[] = {
        'T', 'r', 'a', 'v', 'i', 's', ' ', 
        'B', 'i', 'c', 'k', 'l', 'e', '\0'
    };

    printf("The size of the areas: %ld\n", sizeof(areas));
    printf("Full name='%s'\n", full_name);

    return 0;
}
