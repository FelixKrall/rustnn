# Code coverage

The [Makefile](https://github.com/rustnn/rustnn/blob/main/Makefile) owns the coverage commands
and selected features. Install the prerequisites explicitly:

```bash
cargo install cargo-llvm-cov
rustup component add llvm-tools-preview
```

Then generate a report:

```bash
make coverage         # text summary
make coverage-html    # target/llvm-cov/html/
make coverage-open    # open HTML with macOS `open`; elsewhere open index.html yourself
make coverage-lcov    # target/llvm-cov/lcov.info
```

These targets enable all features and run library unit tests (`--all-features --workspace --lib`);
they do not measure the separate WPT suite.
All-feature builds require the corresponding backend toolchains; use
`cargo llvm-cov --lib --features <features>` directly for a supported subset. Use the generated report
for current line/region coverage and the [WPT dashboard](https://rustnn.github.io/rustnn/wpt-conformance/)
for numerical conformance. Neither metric proves that failure and ownership paths are covered.

When changing an API adapter, inspect its error, buffer-boundary and cleanup paths in addition
to its successful calls. Keep coverage instrumentation separate from performance measurements.
