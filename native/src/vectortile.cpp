#include "vectortile.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <limits>
#include <new>
#include <thread>
#include <vector>

namespace {

struct Candidate {
    double distance;
    int64_t index;
};

// The heap root is the worst retained candidate, including its tie breaker.
struct Better {
    bool operator()(const Candidate& a, const Candidate& b) const noexcept {
        return a.distance < b.distance ||
               (a.distance == b.distance && a.index < b.index);
    }
};

void set_error(char* buffer, size_t capacity, const char* message) noexcept {
    if (buffer == nullptr || capacity == 0) return;
    const size_t count = std::min(capacity - 1, std::strlen(message));
    std::memcpy(buffer, message, count);
    buffer[count] = '\0';
}

int fail(int status, char* buffer, size_t capacity, const char* message) noexcept {
    set_error(buffer, capacity, message);
    return status;
}

// Limit byte offsets as well as element counts before any pointer arithmetic.
bool checked_elements(int64_t rows, int64_t columns, size_t element_size,
                      size_t& result) noexcept {
    if (rows < 0 || columns < 0) return false;
    const auto limit = static_cast<uint64_t>(
        std::min(static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max()),
                 std::numeric_limits<size_t>::max()) / element_size);
    const auto r = static_cast<uint64_t>(rows);
    const auto c = static_cast<uint64_t>(columns);
    if (c != 0 && r > limit / c) return false;
    result = static_cast<size_t>(r * c);
    return true;
}

template <typename T>
bool aligned(const T* pointer) noexcept {
    return reinterpret_cast<uintptr_t>(pointer) % alignof(T) == 0;
}

bool valid_range(const void* pointer, size_t bytes) noexcept {
    return reinterpret_cast<uintptr_t>(pointer) <=
           std::numeric_limits<uintptr_t>::max() - bytes;
}

bool overlaps(const void* a, size_t a_bytes, const void* b, size_t b_bytes) noexcept {
    if (a_bytes == 0 || b_bytes == 0) return false;
    const auto a_start = reinterpret_cast<uintptr_t>(a);
    const auto b_start = reinterpret_cast<uintptr_t>(b);
    return a_start < b_start + b_bytes && b_start < a_start + a_bytes;
}

struct JoinThreads {
    std::vector<std::thread>& threads;
    ~JoinThreads() {
        for (auto& thread : threads) {
            if (thread.joinable()) thread.join();
        }
    }
};

}  // namespace

extern "C" VT_API int vectortile_abi_version(void) { return 1; }

extern "C" VT_API int vectortile_search(
    const double* queries, const double* corpus,
    int64_t query_count, int64_t corpus_count, int64_t dimensions,
    int64_t k, int64_t requested_threads,
    int64_t* indices, double* distances,
    char* error, size_t error_capacity) {
    set_error(error, error_capacity, "");
    try {
        if (query_count < 0 || corpus_count < 1 || dimensions < 1 ||
            k < 1 || k > corpus_count || requested_threads < 1) {
            return fail(1, error, error_capacity, "Invalid shape, k, or thread count.");
        }
        size_t query_elements, corpus_elements, output_elements, heap_elements;
        if (!checked_elements(query_count, dimensions, sizeof(double), query_elements) ||
            !checked_elements(corpus_count, dimensions, sizeof(double), corpus_elements) ||
            !checked_elements(query_count, k, sizeof(double), output_elements) ||
            !checked_elements(1, k, sizeof(Candidate), heap_elements)) {
            return fail(1, error, error_capacity, "Shape or buffer byte size overflows.");
        }
        static_assert(sizeof(double) == sizeof(int64_t), "ABI requires 64-bit double.");
        if (corpus == nullptr || (query_count > 0 &&
            (queries == nullptr || indices == nullptr || distances == nullptr))) {
            return fail(1, error, error_capacity, "Null pointer for a nonempty buffer.");
        }
        if (!aligned(queries) || !aligned(corpus) || !aligned(indices) || !aligned(distances)) {
            return fail(1, error, error_capacity, "Buffers must have natural alignment.");
        }
        const size_t query_bytes = query_elements * sizeof(double);
        const size_t corpus_bytes = corpus_elements * sizeof(double);
        const size_t output_bytes = output_elements * sizeof(double);
        if (!valid_range(queries, query_bytes) || !valid_range(corpus, corpus_bytes) ||
            !valid_range(indices, output_bytes) || !valid_range(distances, output_bytes)) {
            return fail(1, error, error_capacity, "Buffer address range overflows.");
        }
        if (overlaps(indices, output_bytes, distances, output_bytes) ||
            overlaps(indices, output_bytes, queries, query_bytes) ||
            overlaps(indices, output_bytes, corpus, corpus_bytes) ||
            overlaps(distances, output_bytes, queries, query_bytes) ||
            overlaps(distances, output_bytes, corpus, corpus_bytes)) {
            return fail(1, error, error_capacity, "Output buffers must not overlap inputs or each other.");
        }
        for (size_t i = 0; i < query_elements; ++i) {
            if (!std::isfinite(queries[i])) {
                return fail(2, error, error_capacity, "Queries contain a non-finite value.");
            }
        }
        for (size_t i = 0; i < corpus_elements; ++i) {
            if (!std::isfinite(corpus[i])) {
                return fail(2, error, error_capacity, "Corpus contains a non-finite value.");
            }
        }
        if (query_count == 0) return 0;

        const size_t hardware = std::max(1u, std::thread::hardware_concurrency());
        const size_t workers = static_cast<size_t>(std::min<int64_t>(
            std::min<int64_t>(requested_threads, query_count),
            static_cast<int64_t>(std::min<size_t>(hardware, 64))));
        std::atomic<int64_t> next_query{0};
        std::atomic<int> status{0};
        auto record_error = [&](int code) noexcept {
            int expected = 0;
            status.compare_exchange_strong(expected, code, std::memory_order_relaxed);
        };
        auto worker = [&]() noexcept {
            try {
                std::vector<Candidate> heap;
                heap.reserve(static_cast<size_t>(k));
                const Better better;
                while (status.load(std::memory_order_relaxed) == 0) {
                    const int64_t q = next_query.fetch_add(1, std::memory_order_relaxed);
                    if (q >= query_count) break;
                    heap.clear();
                    const double* query = queries + static_cast<size_t>(q) * dimensions;
                    for (int64_t n = 0; n < corpus_count; ++n) {
                        if (status.load(std::memory_order_relaxed) != 0) return;
                        const double* point = corpus + static_cast<size_t>(n) * dimensions;
                        double distance = 0.0;
                        for (int64_t d = 0; d < dimensions; ++d) {
                            const double difference = query[d] - point[d];
                            distance += difference * difference;
                        }
                        // Nonnegative terms mean any overflow remains non-finite.
                        if (!std::isfinite(distance)) {
                            record_error(2);
                            return;
                        }
                        const Candidate candidate{distance, n};
                        if (heap.size() < static_cast<size_t>(k)) {
                            heap.push_back(candidate);
                            std::push_heap(heap.begin(), heap.end(), better);
                        } else if (better(candidate, heap.front())) {
                            std::pop_heap(heap.begin(), heap.end(), better);
                            heap.back() = candidate;
                            std::push_heap(heap.begin(), heap.end(), better);
                        }
                    }
                    std::sort(heap.begin(), heap.end(), better);
                    const size_t offset = static_cast<size_t>(q) * k;
                    for (size_t rank = 0; rank < heap.size(); ++rank) {
                        indices[offset + rank] = heap[rank].index;
                        distances[offset + rank] = heap[rank].distance;
                    }
                }
            } catch (const std::bad_alloc&) {
                record_error(3);
            } catch (...) {
                record_error(4);
            }
        };
        {
            std::vector<std::thread> threads;
            threads.reserve(workers - 1);
            JoinThreads join{threads};
            try {
                for (size_t i = 1; i < workers; ++i) threads.emplace_back(worker);
            } catch (...) {
                record_error(4);
                throw;
            }
            worker();
        }
        const int code = status.load(std::memory_order_relaxed);
        if (code == 2) return fail(2, error, error_capacity, "Squared-L2 distance overflowed float64.");
        if (code == 3) return fail(3, error, error_capacity, "Unable to allocate native search workspace.");
        if (code != 0) return fail(4, error, error_capacity, "Native worker failed.");
        return 0;
    } catch (const std::bad_alloc&) {
        return fail(3, error, error_capacity, "Unable to allocate native search workspace.");
    } catch (...) {
        return fail(4, error, error_capacity, "Unable to start or complete native search.");
    }
}
