# erebus-trainer

A standalone [bullet](https://github.com/jw1912/bullet) trainer that produces an
NNUE checkpoint the **erebus** engine loads byte-for-byte with a one-line change.

- Arch: `(768 -> 1024) x 2 -> 1`, SCReLU, quantised `i16`
- Inputs: bullet `Chess768` (already matches the engine's `feature_index`)
- Engine repo: `/home/jshriver/ocdb/erebus-nnue`, loader: `src/nnue.rs`
- bullet pinned rev: `629ee50000b2afb7b3337595401c830d3b1e0f42`

This is bullet's own `examples/simple.rs` architecture with `HIDDEN_SIZE = 1024`,
plus a standard Stockfish-binpack filter, cosine LR, a linear WDL taper, and
**auto-resume** for interrupted Colab / Kaggle sessions.

Every hyper-parameter is a `const` in the `CONFIG` block at the top of
`src/main.rs` — edit and rebuild. The binary takes exactly one kind of argument:
the data path(s). Resume is automatic and unconditional.

```
cargo build --release --features cuda
./target/release/erebus-trainer data/          # or: a.binpack b.binpack ...
```

---

## Architecture contract (must match `erebus-nnue/src/nnue.rs`)

| thing   | value | engine location |
|---------|-------|-----------------|
| `HL` / `HIDDEN_SIZE` | **1024** | `pub const HL: usize` |
| `QA`    | 255   | `const QA: i64` |
| `QB`    | 64    | `const QB: i64` |
| `SCALE` | 400   | `const SCALE: i64` (= bullet `eval_scale`) |

`QA` / `QB` never change. If you train a different width, change `HIDDEN_SIZE`
in `src/main.rs` **and** `HL` in the engine, and rebuild both.

### Binary layout `parse_net` expects

Little-endian `i16`, concatenated, trailing bytes ignored:

```
l0w : 768 * 1024 = 786,432  i16   feature-major (each feature's 1024 weights contiguous)
l0b : 1024                   i16
l1w : 2 * 1024   = 2,048     i16   first 1024 = stm perspective, next 1024 = ntm
l1b : 1                      i16   (bullet quantises this by QA*QB = 16,320)
```

Meaningful size: `(786432 + 1024 + 2048 + 1) * 2 = 1,579,010 bytes` (~1.5 MB).
bullet pads `quantised.bin` with `"bullet"` to a 64-byte boundary; the loader
ignores it.

The `save_format` in `src/main.rs` produces exactly this:

```rust
SavedFormat::id("l0w").round().quantise::<i16>(QA),
SavedFormat::id("l0b").round().quantise::<i16>(QA),
SavedFormat::id("l1w").round().quantise::<i16>(QB),
SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
```

---

## The one-line engine change

`erebus-nnue/src/nnue.rs`:

```rust
-pub const HL: usize = 64;
+pub const HL: usize = 1024;
```

Nothing else needs to change. Verified against the current loader:

- `Accumulator { v: [i32; HL] }` and every accumulator loop are already
  `HL`-generic — no hardcoded `64`, no fixed SIMD unrolls.
- `evaluate_with_net` accumulates the output dot product in `i64`
  (`sum: i64`, `screlu(_) -> i64`, `l1w[_] as i64`), so widening to 1024 does
  **not** overflow.
- `feature_index` already matches bullet `Chess768`
  (`384*color + 64*piece + sq`, black perspective mirrors color and `sq ^ 56`).

After changing `HL`, the engine will not pass `cargo test --release nnue` until
a real 1024-wide `nets/net.nnue` is dropped in — change the const and deploy the
net together.

---

## Build

Real training needs a GPU backend (this bullet version has **no CPU backend** —
without a feature it compiles against a mock device and will not train).

```bash
# CUDA (Colab, Kaggle, local NVIDIA). Needs the CUDA toolkit / nvcc on PATH.
cargo build --release --features cuda

# ROCm
cargo build --release --features rocm

# no feature: type-checks only, cannot train
cargo check
```

Toolchain: rustc >= 1.87, edition 2024. `rustup update stable` on a fresh
Colab/Kaggle image.

---

## Run

```bash
cargo build --release --features cuda
cp target/release/erebus-trainer  ~/bin/          # reuse the binary anywhere

./erebus-trainer data/                            # a dir of *.binpack (sorted)
./erebus-trainer a.binpack b.binpack c.binpack    # or explicit files
```

That's the whole interface. Everything else is the `CONFIG` block in
`src/main.rs`:

| const | default | note |
|-------|---------|------|
| `HIDDEN_SIZE` | `1024` | must equal `HL` in the engine |
| `NET_ID` | `"erebus"` | checkpoint prefix → `checkpoints/erebus-<N>/`; change it if you train a second width |
| `OUT_DIR` | `"checkpoints"` | put on persistent storage on Colab/Kaggle |
| `END_SUPERBATCH` | `400` | ~40B positions seen (~0.18 epochs of 218B) |
| `BATCH_SIZE` | `16384` | |
| `BATCHES_PER_SUPERBATCH` | `6104` | ~100M positions/superbatch |
| `SAVE_RATE` | `10` | checkpoint every N superbatches |
| `LR_START` / `LR_FINAL` | `0.001` / `2.5e-6` | cosine decay over the run |
| `WDL_START` / `WDL_END` | `0.2` / `0.6` | linear taper: shape early, sharpen late |
| `EVAL_SCALE` | `400` | must equal the engine's `SCALE` |
| `QA` / `QB` | `255` / `64` | never change |
| `CPU_THREADS` | `4` | training-loop workers |
| `DATA_THREADS` | `8` | binpack decode threads |
| `SHUFFLE_BUFFER_MB` | `4096` | bigger = better local mixing |
| `ROTATE_DATA_EACH_SESSION` | `true` | see Resume |

Edit, `cargo build --release --features cuda`, copy the binary, done.

---

## Resume (Colab / Kaggle)

The SF binpack loader has **no data cursor** — every process start reads the
file list from the beginning. Combined with checkpoint resume, that means a
naive restart re-trains on the same early files. This trainer handles it:

1. On start it scans `OUT_DIR` for `<NET_ID>-<N>/` dirs containing
   `optimiser_state/`, takes the highest `N`, calls
   `trainer.load_from_checkpoint()` on it, and sets `start_superbatch = N + 1`.
2. With `ROTATE_DATA_EACH_SESSION = true` it **rotates the data file list left
   by `(start_superbatch - 1) % n_files`** so each resumed session begins on a
   different binpack, spreading coverage.
3. If `N + 1 > END_SUPERBATCH` it prints "already trained" and exits.

So the workflow is just: **re-run the identical command** after a preemption.

```bash
# the cell/script you re-run each session -- OUT_DIR is compiled in
./erebus-trainer /content/drive/MyDrive/erebus/data
```

Set `OUT_DIR` (in `src/main.rs`) to a Google Drive / Kaggle Dataset output path
so checkpoints survive the VM. Each `<NET_ID>-<N>/` is ~15 MB
(`optimiser_state/` + `raw.bin` + `quantised.bin`); 40 of them ≈ 600 MB.

Cosine LR and linear WDL both key off the absolute superbatch index, so they
stay on schedule across any number of resumes.

---

## Data notes

- Public SF binpacks: <https://robotmoon.com/nnue-training-data/>
- Filter applied (standard): `ply >= 16`, side to move not in check,
  `|score| <= 10000`, best move is `Normal` and non-capture.
- `test_set` is **not implemented** in this bullet version (it prints a warning
  and ignores it), so all files go to training. Hold out a couple of binpacks
  manually if you want an offline sanity eval later.
- Files are consumed in list order with a `SHUFFLE_BUFFER_MB`-sized shuffle window,
  then the list repeats. A 400-superbatch run touches roughly the first
  `40B / (positions-per-file)` files before wrapping; order your files (or the
  directory contents) so the early ones are the most diverse. For full
  shuffling across all 41 files, pre-interleave them with bullet's
  `bullet-utils interleave` first.

---

## Deploy into the engine

```bash
# NOTE: current bullet names the file quantised.bin (NOT <net-id>-<N>.bin).
# Deploy quantised.bin ONLY -- raw.bin is unquantised f32 and will load as garbage.
cp checkpoints/erebus-400/quantised.bin  /home/jshriver/ocdb/erebus-nnue/nets/net.nnue

cd /home/jshriver/ocdb/erebus-nnue
#   set `pub const HL: usize = 1024;` in src/nnue.rs (see above)
cargo build --release      # nets/net.nnue is include_bytes!'d, so this embeds it
```

## Validate

```bash
cargo test --release nnue                         # startpos |eval| < 150 ;
                                                  # QQQQ-vs-k > 800, (btm) < -800 ; symmetric small
./target/release/erebus --probe-fen "<fen>"
./target/release/erebus --probe-symmetry "<fen>" "<mirror>"     # sum ~ 0
printf 'position startpos\neval\nquit\n' | ./target/release/erebus
```

Loader failure modes:

| message | cause |
|---------|-------|
| `expected at least N bytes` | `HL` still 64 in the engine, or you copied `raw.bin` / an `optimiser_state` file instead of `quantised.bin` |
| `feature-transformer weights are 0.xxx% non-zero` | untrained / wrong checkpoint |

---

## Later: the wide / threat-aware arch (net 2, not here)

Output buckets, horizontal king mirroring + king buckets, and an f32 L2/L3 tail
each need matching edits to `erebus-nnue/src/nnue.rs` (`feature_index`, the
forward pass, the binary layout). bullet references:
`examples/progression/2_output_buckets.rs`, `3_input_buckets.rs`,
`4_multi_layer.rs`. Keep that as a second binary in this repo when the time
comes; the 218B-position dataset already covers it.
