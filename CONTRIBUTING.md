# Contributing

Use the README quickstart, build the native library, then run the Python and
browser algorithm tests. Changes to distance accumulation, tie handling or
input conversion need independent-oracle regression tests. Do not change
benchmark numbers manually: rerun the script and retain all raw samples.

For numerical comparisons, check squared distances with an explicit tolerance
and reserve exact index comparisons for separated fixtures or deliberate
exact ties. A skipped optional backend is not a passed hardware validation.

Keep inputs synthetic or explicitly licensed. Never commit environment files,
tokens, private models, customer records, or restricted research material.
