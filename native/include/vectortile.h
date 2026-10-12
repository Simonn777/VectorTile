#ifndef VECTORTILE_H
#define VECTORTILE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define VT_API __declspec(dllexport)
#else
#define VT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Contiguous, aligned row-major float64 inputs; Q x k outputs.
 * Buffers must be valid for their declared sizes and outputs must not overlap
 * one another or inputs. Input buffers must remain unchanged during this call.
 * Q may be zero, in which case queries and outputs may be null.
 * Return: 0 success, 1 invalid argument, 2 non-finite input/distance overflow,
 *         3 allocation failure, 4 other runtime failure.
 * error is optional; when error_capacity > 0 it receives a terminated message.
 * No C++ exception is allowed across this boundary.
 */
VT_API int vectortile_search(
    const double* queries, const double* corpus,
    int64_t query_count, int64_t corpus_count, int64_t dimensions,
    int64_t k, int64_t requested_threads,
    int64_t* indices, double* distances,
    char* error, size_t error_capacity);

VT_API int vectortile_abi_version(void);

#ifdef __cplusplus
}
#endif
#endif
