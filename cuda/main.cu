#include <cuda_runtime.h>
#include <iostream>
using namespace std;

int main() {
    int *ptr = NULL;

    printf("1. Initial ptr value: %p\n", (void *)ptr);

    // Check for NULL before using
    if (ptr == NULL) {
        printf("2. ptr is NULL, cannot dereference\n");
    }

    // Allocate memory using cudaMalloc
    if (cudaMalloc((void **)&ptr, sizeof(int)) != cudaSuccess) {
        printf("3. CUDA memory allocation failed\n");
        return 1;
    }

    printf("4. After allocation, ptr value: %p\n", (void *)ptr);

    // Safe to use ptr after NULL check
    int hostValue = 42;
    if (cudaMemcpy(ptr, &hostValue, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("5. Memory copy to device failed\n");
        cudaFree(ptr); // Clean up before exiting
        return 1;
    }

    int retrievedValue;
    if (cudaMemcpy(&retrievedValue, ptr, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("6. Memory copy to host failed\n");
        cudaFree(ptr); // Clean up before exiting
        return 1;
    }

    printf("7. Value retrieved from GPU memory: %d\n", retrievedValue);

    // Clean up
    cudaFree(ptr);
    ptr = NULL;

    printf("8. After free, ptr value: %p\n", (void *)ptr);

    // Demonstrate safety of NULL check after free
    if (ptr == NULL) {
        printf("9. ptr is NULL, safely avoided use after free\n");
    }

    return 0;

}
