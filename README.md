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
cargo build --release
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

The `cuda` feature is **on by default** — this bullet version has no CPU
backend, and the trainer only runs on NVIDIA hardware. The build needs the CUDA
toolkit (`CUDA_PATH` set), which Colab/Kaggle GPU images already have.

```bash
cargo build --release                        # CUDA build

cargo build --release --no-default-features --features rocm   # ROCm instead
cargo check  --no-default-features           # syntax-only, no toolkit needed
```

Toolchain: rustc >= 1.87, edition 2024. `rustup update stable` on a fresh
Colab/Kaggle image.

---

## Run

Training is planned as **one global schedule** over the whole corpus (every file
in `POSITION_COUNTS`): `TOTAL_PASSES` epochs → `GLOBAL_END` superbatches, with the
cosine LR and linear WDL running over `1..=GLOBAL_END`. Each invocation resumes
from the last checkpoint and trains **one pass over the file(s) you pass it**,
then stops. Sessions advance along the one schedule — no per-file LR restarts.

```bash
cargo build --release
cp target/release/erebus-trainer  ~/bin/

# A) all files on disk at once -- best mixing, one pass then it stops
./erebus-trainer data/                       # re-run after any preemption

# B) all files on disk, one per session in a reshuffled order
./run-all.sh data/                           # resumable; self-stops at the plan end

# C) only room for one binpack at a time -- download, train, delete, next
DL_CMD='curl -fL --retry 5 -o "$NAME" "https://HOST/PATH/$NAME"' \
  ./train-staged.sh /mnt/scratch/binpacks    # walks train-order.txt; resumable
```

`train-order.txt` is the 41 basenames in a fixed shuffled order (regenerate with
`shuf` if you like — just keep it fixed for the whole run). `train-staged.sh`
fetches each via your `$DL_CMD`, trains one session, deletes the binpack, and
records a marker so a re-run continues where it stopped.

Exit codes: `0` a session trained, `3` nothing to do (plan complete), `1` error.

Everything else is the `CONFIG` block in `src/main.rs`:

| const | default | note |
|-------|---------|------|
| `HIDDEN_SIZE` | `1024` | must equal `HL` in the engine |
| `NET_ID` | `"erebus"` | checkpoint prefix → `checkpoints/erebus-<N>/`; change it if you train a second width |
| `OUT_DIR` | `"checkpoints"` | put on persistent storage on Colab/Kaggle |
| `TOTAL_PASSES` | `1.0` | epochs over the whole corpus → `GLOBAL_END = round(TOTAL_PASSES * Σ POSITION_COUNTS * FILTER_KEEP_FRAC / (BATCHES_PER_SUPERBATCH * BATCH_SIZE))` = 2188 at `1.0`. Raise & rebuild to train longer. |
| `PASS_FRACTION_PER_FILE` | `1.0` | how much of each passed file one session consumes (`session_end = min(resume + round(PASS_FRACTION_PER_FILE * passed_positions * FILTER_KEEP_FRAC / pos_per_sb), GLOBAL_END)`) |
| `POSITION_COUNTS` | 41 rows | compiled-in `(basename, raw_count)` table; per-file `passed_positions` source (a `<file>.binpack.count` sidecar, plain integer, overrides it) |
| `FILTER_KEEP_FRAC` | `1.0` | est. fraction of raw positions surviving `filter()`; lower to ~0.6 if the loader wraps before a session ends |
| `MAX_SUPERBATCH` | `5000` | safety cap on a single **session** budget (not `GLOBAL_END`) |
| `EREBUS_END_SUPERBATCH=N` | — | env override: force this session to end at superbatch N |
| `BATCH_SIZE` | `16384` | |
| `BATCHES_PER_SUPERBATCH` | `6104` | ~100M positions/superbatch |
| `SAVE_RATE` | `10` | checkpoint every N superbatches |
| `LR_START` / `LR_FINAL` | `0.001` / `2.5e-6` | cosine decay over `1..=GLOBAL_END` |
| `WDL_START` / `WDL_END` | `0.2` / `0.6` | linear taper over `1..=GLOBAL_END`: shape early, sharpen late |
| `EVAL_SCALE` | `400` | must equal the engine's `SCALE` |
| `QA` / `QB` | `255` / `64` | never change |
| `CPU_THREADS` | `4` | training-loop workers |
| `DATA_THREADS` | `8` | binpack decode threads |
| `SHUFFLE_BUFFER_MB` | `4096` | bigger = better local mixing |
| `ROTATE_DATA_EACH_SESSION` | `true` | rotate a multi-file list per session (no-op with one file/session) |

Edit, `cargo build --release`, copy the binary, done.

---

## Resume (Colab / Kaggle)

The SF binpack loader has **no data cursor** — every process start reads the
file list from the beginning. Combined with checkpoint resume, that means a
naive restart re-trains on the same early files. This trainer handles it:

1. On start it scans `OUT_DIR` for `<NET_ID>-<N>/` dirs containing
   `optimiser_state/`, takes the highest `N`, calls
   `trainer.load_from_checkpoint()` on it, and sets `start_superbatch = N + 1`.
2. `OUT_DIR/<NET_ID>.session` holds `<began_at> <stop_at>` for the current
   session. If the resume point falls inside that window the session **finishes
   the same window** instead of opening a fresh one from the (later) resume
   point — so a preempted session resumes exactly. It's deleted on clean finish.
3. With `ROTATE_DATA_EACH_SESSION = true` a multi-file list is rotated left by
   `(start_superbatch - 1) % n_files` (no effect with one file per session).
4. If `start_superbatch > GLOBAL_END` it prints the plan is complete and exits 3.

So the workflow is just: **re-run the identical command / `run-all.sh`** after a
preemption.

```bash
# re-run each session -- OUT_DIR is compiled in
./erebus-trainer /content/drive/MyDrive/erebus/data      # all files at once, or
./run-all.sh     /content/drive/MyDrive/erebus/data      # one per session, shuffled
```

Set `OUT_DIR` (in `src/main.rs`) to a Google Drive / Kaggle Dataset output path
so checkpoints survive the VM. Each `<NET_ID>-<N>/` is ~15 MB
(`optimiser_state/` + `raw.bin` + `quantised.bin`); 40 of them ≈ 600 MB.

Cosine LR and linear WDL both run over the absolute `1..=GLOBAL_END` window, so
they stay on schedule across any number of resumes and session splits.

---

## Data notes

- Public SF binpacks: <https://robotmoon.com/nnue-training-data/>
- Filter applied (standard): `ply >= 16`, side to move not in check,
  `|score| <= 10000`, best move is `Normal` and non-capture.
- `test_set` is **not implemented** in this bullet version (it prints a warning
  and ignores it), so all files go to training. Hold out a couple of binpacks
  manually if you want an offline sanity eval later.
- Files are consumed in list order with a `SHUFFLE_BUFFER_MB`-sized shuffle
  window, then the list repeats. **Passing all files at once** (`erebus-trainer
  data/`) mixes every source within each superbatch — strongly preferred if disk
  allows. **One file per session** is sequential across the run, so **the order
  you feed files matters**: download/feed the 41 in a randomised order (fix it
  once and keep it), not month-by-month, or a net that plateaus early will only
  have seen the first few sources. `run-all.sh` does this shuffling for you when
  the files are all present; with on-demand staging, shuffle your download list.
  For the strongest run, pre-interleave all 41 with `bullet-utils interleave`.

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
