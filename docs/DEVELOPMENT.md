# Development provenance

VectorTile is an independent educational systems project created in October
2026 in Wumou Xie's GitHub portfolio. Implementation, tests, documentation and
the browser demonstration were developed with substantial AI assistance from
OpenAI Codex, including parallel implementation and review passes.

The repository is intentionally inspectable: the source, regression tests,
native build configuration, raw timing samples and limitations are public.
Commits reflect the actual development session; no history or pre-existing
experience is simulated. Repository ownership is not a claim of unaided
authorship or prior production deployment.

## Scope and source material

- New, standalone implementation using C++17, Python, NumPy and optional
  PyTorch. No private research code, pretrained weights, customer data, course
  assignments or restricted datasets are included.
- Benchmark vectors are generated from documented NumPy seeds. The browser
  visualization uses its own deterministic 2D generator and JavaScript search.
- The problem is conventional exhaustive top-k retrieval. The project claims
  an explained engineering implementation, not a new search algorithm or a
  state-of-the-art retrieval result.
- This is not affiliated with or endorsed by NVIDIA. CUDA backend availability
  is not evidence of tested performance on NVIDIA hardware.

## Validation boundaries

Python/C++ tests check the API against independent direct-distance oracles;
the C++ smoke harness also runs with address and undefined-behavior sanitizers.
GitHub Actions publishes current cross-platform results. Optional CUDA tests
are explicitly skipped where the runtime or hardware is absent.

The checked-in CPU report records a particular Apple Silicon machine, software
versions, compiler flags, source hashes, seeds and every timing sample.
Measured results are not guarantees for other hardware or datasets. Analytical
workspace counts are not measured peak resident memory.

## Design documentation

- [Backend algorithms and memory accounting](BACKENDS.md)
- [C++ memory, ABI, concurrency and error handling](../native/README.md)
- [Reproduction and interpretation](BENCHMARKS.md)
- [Engineering decisions](DESIGN.md)
