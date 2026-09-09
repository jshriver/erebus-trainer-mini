# Engine update: adopt the HL=1024 net from erebus-trainer

Paste this into a Claude session **in the `erebus-nnue` repo**. It describes the
only change the engine needs to load nets produced by `erebus-trainer`
(github: this repo's sibling, `/home/jshriver/ocdb/erebus-trainer-mini`).

---

## TL;DR

The trainer builds bullet's `examples/simple.rs` architecture — the *same* one
`src/nnue.rs` already implements — but **1024 wide instead of 64**. The engine
change is a single constant:

```rust
// src/nnue.rs
-pub const HL: usize = 64;
+pub const HL: usize = 1024;
```

Then drop in the new weights and rebuild:

```bash
cp <trainer>/checkpoints/erebus-<N>/quantised.bin  nets/net.nnue
cargo build --release
```

`nets/net.nnue` is `include_bytes!`'d, so the rebuild embeds the new net.

---

## Why nothing else in `src/nnue.rs` changes

The architecture, quantisation, feature indexing, and binary layout are all
identical to what `parse_net` / `evaluate_with_net` / `feature_index` already do.
Only the width differs. Verified against the current loader:

1. **`parse_net` size check is already `HL`-derived.**
   `expected = (INPUT_SIZE*HL + HL + 2*HL + 1) * 2`. With `HL = 1024` this
   becomes `(786432 + 1024 + 2048 + 1) * 2 = 1_579_010` bytes automatically. The
   feature-major layout (`l0w` `[768][HL]`, `l0b` `[HL]`, `l1w` `[2*HL]` = stm
   half then ntm half, `l1b` `[1]`, trailing `"bullet"` padding ignored) is
   unchanged.

2. **The accumulator is already `HL`-generic.**
   `Accumulator { v: [i32; HL] }` and every `for h in 0..HL` loop
   (`refresh_one`, `toggle_feature`, `evaluate_with_net`) scale with the const.
   No hardcoded `64`, no fixed-size arrays other than `[_; HL]`, no manual SIMD
   unroll that assumes a width.

3. **The output dot product cannot overflow at 1024.**
   `evaluate_with_net` accumulates in `i64` (`let mut sum: i64 = 0`),
   `screlu(x) -> i64`, and multiplies by `net.l1w[h] as i64`. Per-element
   magnitudes are unchanged by width (each accumulator element is still a sum of
   ≤32 `i16` weight columns), only the term count doubles (2×1024). Well within
   `i64`. `l1b` stays `i32` in the struct and is widened to `i64` in the final
   expression — fine.

4. **`feature_index` already matches bullet `Chess768`.**
   White: `384*color + 64*piece + sq`. Black: `384*(color ^ 1) + 64*piece +
   (sq ^ 56)`. Piece order Pawn..King = 0..5, colour White/Black = 0/1, square
   `rank*8 + file`. The trainer uses `Chess768` unmodified, which is exactly
   this. Colour-fixed (white-persp / black-persp) accumulators with stm/ntm
   selection at eval time is equivalent to bullet's stm/ntm training.

5. **Constants match.** `QA = 255`, `QB = 64`, `SCALE = 400`. The trainer's
   `save_format` quantises `l0w`/`l0b` by `QA`, `l1w` by `QB`, `l1b` by
   `QA*QB` (= 16320). `EVAL_SCALE = 400` in the trainer == `SCALE` here. Do not
   change `QA`/`QB` ever; only change `SCALE` if you also change it in the
   trainer's `EVAL_SCALE`.

6. **Nothing outside `src/nnue.rs` is affected.** The `king_sq` / `piece_count`
   params threaded through `board.rs` / `probe.rs` are still accepted-and-ignored
   (no king buckets, no output buckets in this arch). `update_threats` stays a
   no-op. The runtime `load()` path (`--eval-file` / `setoption EvalFile`) uses
   the same `parse_net`, so it also just works.

### Also update (cosmetic, not functional)

The module doc comment at the top of `src/nnue.rs` states `HL = 64` in a couple
of places (the architecture sketch and the `NET SOURCE` paragraph). Update those
to `1024` so the comment doesn't lie. No code impact.

---

## Forward pass (reference, unchanged)

```
sum   = Σ_{h<HL} screlu(us.v[h]) * l1w[h]  +  Σ_{h<HL} screlu(them.v[h]) * l1w[HL+h]
screlu(x) = clamp(x, 0, QA)^2                       // QA = 255
eval_cp   = (sum / QA + l1b) * SCALE / (QA * QB)    // SCALE = 400, QB = 64
```

`us` = side-to-move accumulator, `them` = the other. `l1b` = raw `i16` from the
file.

---

## Validate after deploying

```bash
cargo test --release nnue
#   embedded_net_parses                      -> passes (right size + non-zero FT weights)
#   startpos_eval_is_small_and_symmetric_ish -> |eval| < 150
#   massive_material_advantage_is_detected   -> QQQQ-vs-k > 800 ; (btm) < -800
#   color_flip_symmetry                      -> both evals < 200

./target/release/erebus --probe-fen "<fen>"
./target/release/erebus --probe-symmetry "<fen>" "<mirror>"     # sum ~ 0
printf 'position startpos\neval\nquit\n' | ./target/release/erebus
```

Between changing `HL` and dropping in a real 1024-wide `nets/net.nnue`, the
`nnue` tests will fail — the shipped 64-wide file is now the wrong size. Change
the const and deploy the net in the same commit.

### Loader failure modes

| message | cause |
|---|---|
| `net data is N bytes, expected at least 1579010 …` | `HL` still 64, **or** you copied `raw.bin` / an `optimiser_state/` file instead of `quantised.bin` |
| `feature-transformer weights are 0.xxx% non-zero` | untrained / wrong checkpoint |
| startpos eval huge, or QQQQ test fails | wrong `SCALE`, or `l1w` halves swapped (stm/ntm) — but layout is unchanged, so suspect a bad checkpoint |

---

## What the trainer produces

- Path: `<trainer>/checkpoints/erebus-<N>/quantised.bin` where `<N>` is the
  superbatch number (final is `erebus-400` at default settings).
- `quantised.bin` is ~1.58 MB: `768*1024` + `1024` + `2*1024` + `1` `i16`
  values, little-endian, in that order, plus `"bullet"` padding to a 64-byte
  boundary.
- **Deploy `quantised.bin` only.** The sibling `raw.bin` is unquantised `f32`
  and is larger, so it passes the size check and then evaluates to noise.

---

## Later (not this change)

Output buckets, horizontal king mirroring + king buckets, and an f32 L2/L3 tail
each require real edits here — `feature_index`, the forward pass, and the binary
layout. That's "net 2"; the trainer repo is set up to host a second binary for
it. This change is width-only.
