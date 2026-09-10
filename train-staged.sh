#!/usr/bin/env bash
#
# One-binpack-at-a-time driver: for each file in train-order.txt, download it,
# run one training session on it, delete it, move on. Resumes cleanly after any
# interruption (finished files are marked; a cut-off session resumes via the
# trainer's own checkpoint + .session file, re-downloading that one binpack).
#
# You supply the download step via $DL_CMD, a shell snippet run with $NAME set
# to the binpack basename and cwd = $DATA_DIR. Examples:
#   export DL_CMD='curl -fL --retry 5 -o "$NAME" "https://HOST/PATH/$NAME"'
#   export DL_CMD='rclone copy "remote:binpacks/$NAME" .'
#   export DL_CMD='python3 /path/fetch.py "$NAME"'
#
# Usage:
#   DL_CMD='curl -fL -o "$NAME" "https://.../$NAME"' \
#     ./train-staged.sh /mnt/scratch/binpacks
#
# Keep OUT_DIR (checkpoints/, set in src/main.rs) on persistent storage.

set -euo pipefail

DATA_DIR="${1:?usage: train-staged.sh <staging-dir>  (with \$DL_CMD set)}"
TRAINER="${TRAINER:-$PWD/target/release/erebus-trainer}"
ORDER="${ORDER:-$PWD/train-order.txt}"
OUT_DIR="${EREBUS_OUT_DIR:-$PWD/checkpoints}"
STATE_DIR="$OUT_DIR/.staged"
KEEP_BINPACK="${KEEP_BINPACK:-0}"     # set 1 to not delete after training

: "${DL_CMD:?set DL_CMD to a snippet that fetches \$NAME into the cwd}"
[ -x "$TRAINER" ] || { echo "trainer not executable: $TRAINER" >&2; exit 1; }
[ -f "$ORDER" ]   || { echo "order file not found: $ORDER" >&2; exit 1; }
mkdir -p "$DATA_DIR" "$STATE_DIR"

while IFS= read -r NAME; do
    [ -n "$NAME" ] || continue
    case "$NAME" in \#*) continue ;; esac

    marker="$STATE_DIR/$NAME.done"
    if [ -f "$marker" ]; then
        echo "skip (done): $NAME"
        continue
    fi

    f="$DATA_DIR/$NAME"
    if [ ! -s "$f" ]; then
        echo "=== fetching $NAME ==="
        ( cd "$DATA_DIR" && eval "$DL_CMD" )
        [ -s "$f" ] || { echo "download produced no $f" >&2; exit 1; }
    fi

    echo "=== training on $NAME ==="
    rc=0
    "$TRAINER" "$f" || rc=$?
    case "$rc" in
        0) touch "$marker"
           [ "$KEEP_BINPACK" = 1 ] || rm -f "$f"
           ;;
        3) echo "trainer: plan complete -- stopping."
           [ "$KEEP_BINPACK" = 1 ] || rm -f "$f"
           exit 0
           ;;
        *) echo "trainer exited $rc on $NAME -- stopping. re-run to resume." >&2
           exit "$rc"
           ;;
    esac
done < "$ORDER"

echo "all $(grep -cvE '^\s*(#|$)' "$ORDER") files trained (or plan completed earlier)."
