#include <stdio.h>
#include <stdint.h>

typedef struct {
    float x;
    float y;
    float z;
} Point;

int main() {
    // execute_example1();
    // example2();

    custom_types();
}

void custom_types() {

    Point p = {1.45, 4.5};
    printf("size of point : %zu\n", sizeof(Point));

}

void example2() {

    int arr1[] = {1, 2, 3, 4};
    int arr2[] = {5, 6, 7, 8};

    int *ptr1 = arr1, *ptr2 = arr2;
    int *mat[] = {ptr1, ptr2};

    // printf("%a", *mat);
    // printf("Done");

    printf("Checking the printed form : \n");
    for (int i=0; i<2; i++) {
        for (int j=0; j<(sizeof(arr1)/sizeof(arr1[0])); j++) {
            printf("%p ", mat[i]);
            printf("%d ", *mat[i]);
            mat[i]++;
        }
        printf("\n");
    }
}

void execute_example1() {

    int arr[] = {12, 14, 16, 18, 20};
    int* ptr = arr;

    printf("Position One : %d\n", *ptr);
    int len = sizeof(arr)/sizeof(arr[0]);

    printf("Size of the arr : %d\n", len);
    for (int i=0; i<len; i++) {
        printf("%d ", *ptr);
        print_binary(ptr);
        printf("%p\n", ptr);
        ptr++;
    }

    printf("Try and print the pointer at the end as well : %d %p\n", *ptr, ptr);
}

void print_binary(int* ptr) {

    uintptr_t addr = (uintptr_t)ptr;
    for (int i = (sizeof(addr) * 8) - 1; i >= 0; --i) {
        printf("%d", (addr >> i) & 1);
        if (i % 4 == 0) printf(" "); 
    }
    printf("\n");
}
