#ifndef TENSOR_TEST_HELPERS_H
#define TENSOR_TEST_HELPERS_H

#include <stdio.h>
#include <stdlib.h>

/**
 * Test assertion macro that is guaranteed to always evaluate the expression,
 * regardless of NDEBUG or compiler optimization flags.
 */
#define TEST_ASSERT(expr) do { \
    if (!(expr)) { \
        fprintf(stderr, "[FAIL] Assertion failed: %s (%s:%d)\n", #expr, __FILE__, __LINE__); \
        abort(); \
    } \
} while (0)

#endif /* TENSOR_TEST_HELPERS_H */
