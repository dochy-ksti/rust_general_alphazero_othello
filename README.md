# Rust_General_Alphazero_Othello

Reimplementation of [Alpha Zero General](https://github.com/suragnair/alpha-zero-general) in Rust. This only includes the implementation of the Othello game.

Supports concurrent self-play and arena comparison. Roughly 15 times faster on RTX 3060(It should depend on the horsepower of your GPU).

This needs CUDA and pytorch. Rust and Python are communicating using a very primitive C FFI.

"cargo build" to build the CDLL, and "python -m python.main" to start learning. If you want to use the release build, "cargo build --release" and "python -m python.main release"