# include <stdio.h>

int main(int argc, char *argv[])
{
    int numbers[4] = {1, 2, 3, 4};
    char name[4] = {'a', 'b'};

    printf(
        "expected:\n\t");
    printf(
        "numbers: 1 2 3 4\n");
    printf(
        "numbers: %d %d %d %d\n",
        numbers[0], numbers[1],
        numbers[2], numbers[3]);

    printf(
        "expected:\n\t");
    printf(
        "name each: a b\n");
    printf(
        "name each: %c %c %c %c\n",
        name[0], name[1],
        name[2], name[3]);

    name[0] = 'L';
    name[1] = 'e';
    name[2] = 'x';
    name[3] = '\0';

    printf("name: %s\n", name);

    // Another way to use name
    char *another = "Travis";
    printf("another: %s\n", another);

    // We should see an error here
    name[3] = "F";
    printf("name: %s\n", name);
}
