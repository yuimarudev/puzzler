# PuzzleCaptchaSolver

inspired by [vsmutok/PuzzleCaptchaSolver](https://github.com/vsmutok/PuzzleCaptchaSolver)

## How to use

```rust
use puzzler::PuzzleCaptchaSolver;

fn main() -> Result<(), puzzler::PuzzlePatternError> {
  let solver = PuzzleCaptchaSolver::new(
    "/path/to/slice.png",
    "/path/to/background.png",
    "/path/to/result.png",
  );
  let x = solver.discern()?;

  println!("{x}");

  Ok(())
}
```


Would you like to know about other APIs? I've included doc comments, so generate them using `cargo doc`

## Performance

The numbers below compare the Rust implementation with the original Python implementation on the repository sample images:

- gap image: `./slice.png`
- background image: `./bg.png`
- expected position: `136`
- rounds: `1000`
- measurement style: repeated calls in one process

The comparison target was `discern()`, because it matches the Python program's full flow:

1. load images
2. crop whitespace from the gap image
3. build feature images
4. search for the best position
5. write the marked result image

| implementation | position | total time | average per call |
|---|---:|---:|---:|
| Rust `PuzzleCaptchaSolver::discern()` | 136 | 1131.173 ms | 1.131 ms |
| Python `cv2.matchTemplate` version | 136 | 8382.297 ms | 8.382 ms |

On this machine, the Rust implementation was about `7.41x` faster for this input.

Commands used:

```bash
cargo run --release --bin puzzler_bench -- ./slice.png ./bg.png result_rust.png 136 1000 1000
```

## Benchmark environment

- OS: `Linux 7.0.0-1-cachyos #1 SMP PREEMPT Wed, 15 Apr 2026 06:34:45 +0000 x86_64 GNU/Linux` (Arch Linux with cachyos zen4 kernel)
- CPU: `AMD Ryzen 7 7840U w/ Radeon 780M Graphics`
- CPU topology: `8` cores / `16` threads
- Memory: `59 GiB`
- Rust: `rustc 1.95.0-nightly (1ed488274 2026-02-25)`
- LLVM: `22.1.0`
- Python: `3.11.14`
- OpenCV for Python: `4.13.0`

Cargo release settings:

- `opt-level = 3`
- `lto = "fat"`
- `codegen-units = 1`
- `strip = true`
- `debug = false`
- `incremental = false`
- `panic = "unwind"`

The machine may have had unrelated build load while the benchmark was running, so the absolute time can move. The comparison above was taken under the same host state for both implementations.
