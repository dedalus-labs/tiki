// Calls the guarded store with an offset that wraps a 32-bit integer.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void store(float *base, int32_t row, int32_t col, int32_t len, float value);

int main(void) {
    enum { length = 16 };
    float *buffer = calloc(length, sizeof(float));
    store(buffer, 32768, 0, length, 1.0f);
    printf("guard rejected the store\n");
    free(buffer);
    return 0;
}
