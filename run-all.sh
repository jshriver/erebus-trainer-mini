#!/usr/bin/env bash
#
# Train erebus over every *.binpack in a directory, one file per training
# session, in a freshly shuffled order each pass, resuming from the last
# checkpoint every time. The compiled-in TOTAL_PASSES (src/main.rs) decides
# when the net is done -- this script just keeps feeding files until the
# trainer reports the plan is complete (exit code 3).
#
# Safe to re-run after any interruption (Colab/Kaggle preemption, Ctrl-C):
#   - per-pass file order is persisted, so order is stable across restarts
#   - finished files are skipped via markers
#   - a file cut off mid-session resumes exactly (trainer + its .session file)
#
# Usage:
#   ./run-all.sh <dir-with-binpacks> [path/to/erebus-trainer]
#
# State lives in <OUT_DIR>/.run-all/ (OUT_DIR defaults to "checkpoints", matching
# src/main.rs -- override with EREBUS_OUT_DIR if you changed it there).

set -euo pipefail

DATA_DIR="${1:?usage: run-all.sh <dir-with-binpacks> [erebus-trainer-binary]}"
TRAINER="${2:-./target/release/erebus-trainer}"
OUT_DIR="${EREBUS_OUT_DIR:-checkpoints}"
STATE_DIR="$OUT_DIR/.run-all"
MAX_PASSES="${EREBUS_MAX_PASSES:-10}"   # hard stop; TOTAL_PASSES normally ends it first

[ -x "$TRAINER" ] || { echo "trainer not found/executable: $TRAINER" >&2; exit 1; }
[ -d "$DATA_DIR" ] || { echo "not a directory: $DATA_DIR" >&2; exit 1; }
mkdir -p "$STATE_DIR"

mapfile -t ALL < <(find "$DATA_DIR" -maxdepth 1 -type f -name '*.binpack' | sort)
[ "${#ALL[@]}" -gt 0 ] || { echo "no *.binpack files in $DATA_DIR" >&2; exit 1; }
echo "found ${#ALL[@]} binpacks in $DATA_DIR"

for ((pass = 1; pass <= MAX_PASSES; pass++)); do
    order="$STATE_DIR/pass-$pass.order"
    if [ ! -f "$order" ]; then
        printf '%s\n' "${ALL[@]}" | shuf > "$order"
    fi
    echo "======== pass $pass ($(wc -l < "$order") files) ========"

    while IFS= read -r f; do
        base=$(basename "$f")
        done_marker="$STATE_DIR/pass-$pass.$base.done"
        if [ -f "$done_marker" ]; then
            echo "  skip (done): $base"
            continue
        fi

        echo "  --- $base ---"
        rc=0
        "$TRAINER" "$f" || rc=$?
        case "$rc" in
            0) touch "$done_marker" ;;
            3) echo "trainer reports plan complete -- stopping."; exit 0 ;;
            *) echo "trainer exited $rc on $base -- stopping (re-run to resume)." >&2; exit "$rc" ;;
        esac
    done < "$order"

    echo "======== pass $pass complete ========"
done

echo "reached EREBUS_MAX_PASSES=$MAX_PASSES without a 'plan complete' -- check TOTAL_PASSES." >&2
exit 1
