// Standalone harness: can run with ASan/UBSan without instrumenting Python.
#include "vectortile.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

int main() {
    assert(vectortile_abi_version() == 1);
    const double queries[] = {0, 0, 0, 0, 0, 0};
    const double corpus[] = {1, 0, -1, 0, 0, 0, 0, 1};
    int64_t indices[9]{};
    double distances[9]{};
    char error[128]{};
    int status = vectortile_search(queries, corpus, 3, 4, 2, 3, 10000,
                                  indices, distances, error, sizeof(error));
    assert(status == 0);
    for (int row = 0; row < 3; ++row) {
        assert(indices[row * 3] == 2);
        assert(indices[row * 3 + 1] == 0);
        assert(indices[row * 3 + 2] == 1);
        assert(distances[row * 3] == 0);
        assert(distances[row * 3 + 1] == 1);
    }
    status = vectortile_search(nullptr, corpus, 0, 4, 2, 3, 1,
                              nullptr, nullptr, error, sizeof(error));
    assert(status == 0);
    status = vectortile_search(queries, corpus, INT64_MAX, 4, 2, 3, 1,
                              indices, distances, error, sizeof(error));
    assert(status == 1);
    status = vectortile_search(nullptr, nullptr, 1, 1, 1, 1, 1,
                              nullptr, nullptr, error, sizeof(error));
    assert(status == 1);
    const double nan[] = {std::numeric_limits<double>::quiet_NaN()};
    status = vectortile_search(nan, corpus, 1, 1, 1, 1, 1,
                              indices, distances, error, sizeof(error));
    assert(status == 2);
    const double huge[] = {1e200};
    status = vectortile_search(huge, corpus, 1, 1, 1, 1, 1,
                              indices, distances, error, sizeof(error));
    assert(status == 2);
    return 0;
}
