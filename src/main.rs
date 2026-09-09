//! erebus-trainer -- bullet trainer for the erebus "simple" NNUE arch.
//!
//!     (768 -> HIDDEN_SIZE) x 2 -> 1,  SCReLU,  quantised i16
//!
//! Produces `<OUT_DIR>/<NET_ID>-<N>/quantised.bin`, which the erebus engine
//! loads byte-for-byte via `src/nnue.rs::parse_net` once its `HL` const is set
//! to HIDDEN_SIZE below. See README.md for the full contract and deploy steps.
//!
//! Usage:
//!     cargo build --release --features cuda
//!     ./erebus-trainer <data.binpack | data-dir> [more paths...]
//!
//! Every training hyper-parameter lives in the CONFIG block below -- edit it and
//! rebuild. The only runtime argument is where the data is.
//!
//! Resume is automatic: on start it looks for `<OUT_DIR>/<NET_ID>-<N>/` and, if
//! found, loads the highest N and continues from superbatch N+1. Re-run the same
//! command after a Colab/Kaggle preemption and it picks up where it stopped.

use bullet_lib::{
    game::inputs::Chess768,
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{TrainingSchedule, TrainingSteps, lr, wdl},
        settings::LocalSettings,
    },
    value::{ValueTrainerBuilder, loader},
};
use loader::sfbinpack::{MoveType, PieceType, SfBinpackLoader, TrainingDataEntry};

// ========================= CONFIG -- edit, then `cargo build --release` =========================

/// Hidden / accumulator width per perspective. MUST equal `HL` in the engine's
/// `src/nnue.rs`. Changing this is a recompile of BOTH this trainer and erebus.
const HIDDEN_SIZE: usize = 1024;

/// Checkpoint name prefix. Checkpoints land in `<OUT_DIR>/<NET_ID>-<N>/`.
/// Change this (e.g. to "erebus-512") if you ever train a second width, so the
/// runs don't resume into each other.
const NET_ID: &str = "erebus";
/// Where checkpoints are written / resumed from. Put this on persistent storage
/// (Google Drive, a Kaggle Dataset output) when on a preemptible VM.
const OUT_DIR: &str = "checkpoints";

/// Total training length in superbatches.
const END_SUPERBATCH: usize = 400;
/// Positions per batch.
const BATCH_SIZE: usize = 16_384;
/// Batches per superbatch. 6104 * 16384 ~= 100M positions / superbatch.
const BATCHES_PER_SUPERBATCH: usize = 6104;
/// Save a checkpoint every this many superbatches (also always saves the last).
const SAVE_RATE: usize = 10;

/// Learning rate: cosine decay from LR_START to LR_FINAL over END_SUPERBATCH.
const LR_START: f32 = 0.001;
const LR_FINAL: f32 = 2.5e-6;

/// WDL lambda: linear taper from WDL_START (superbatch 1) to WDL_END (last).
/// target = lambda * game_result + (1 - lambda) * sigmoid(score / EVAL_SCALE)
const WDL_START: f32 = 0.2;
const WDL_END: f32 = 0.6;

/// Eval scale. MUST equal the engine's `const SCALE`. Also bullet's eval_scale.
const EVAL_SCALE: i32 = 400;
/// Feature-transformer quantisation. Engine: `const QA: i64 = 255;`
const QA: i16 = 255;
/// Output-weight quantisation. Engine: `const QB: i64 = 64;`
const QB: i16 = 64;

/// CPU worker threads for the training loop.
const CPU_THREADS: usize = 4;
/// Threads for decoding SF binpacks.
const DATA_THREADS: usize = 8;
/// Shuffle-buffer size in MiB. Bigger = better local mixing before batching.
const SHUFFLE_BUFFER_MB: usize = 4096;

/// bullet's SF loader has no data cursor: every process start reads the file
/// list from the beginning. When true, the list is rotated left by the resume
/// superbatch so each resumed session begins on a different binpack.
const ROTATE_DATA_EACH_SESSION: bool = true;

// =============================================================================================

fn die(msg: impl AsRef<str>) -> ! {
    eprintln!("erebus-trainer: {}", msg.as_ref());
    std::process::exit(1);
}

fn usage() -> ! {
    eprintln!(
        "erebus-trainer -- bullet trainer for erebus (768 -> {HIDDEN_SIZE})x2 -> 1 NNUE\n\
         \n\
         usage:  erebus-trainer <PATH>...\n\
         \n\
         each PATH is a .binpack file or a directory (all *.binpack inside it,\n\
         sorted by name). All hyper-parameters are compiled in -- see the CONFIG\n\
         block in src/main.rs. Resume from {OUT_DIR}/{NET_ID}-<N>/ is automatic."
    );
    std::process::exit(2);
}

/// Expand args into a concrete list of .binpack file paths.
fn collect_data_paths(inputs: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for inp in inputs {
        let p = std::path::Path::new(inp);
        if p.is_dir() {
            let mut found: Vec<String> = std::fs::read_dir(p)
                .unwrap_or_else(|e| die(format!("read_dir {inp}: {e}")))
                .flatten()
                .map(|e| e.path())
                .filter(|q| q.extension().map(|x| x == "binpack").unwrap_or(false))
                .map(|q| q.to_string_lossy().into_owned())
                .collect();
            found.sort();
            if found.is_empty() {
                die(format!("no *.binpack files in directory {inp}"));
            }
            out.extend(found);
        } else if p.is_file() {
            out.push(inp.clone());
        } else {
            die(format!("data path does not exist: {inp}"));
        }
    }
    out
}

/// Highest N such that `<OUT_DIR>/<NET_ID>-<N>/optimiser_state/` exists.
fn latest_checkpoint() -> Option<(String, usize)> {
    let prefix = format!("{NET_ID}-");
    let mut best: Option<(String, usize)> = None;
    for ent in std::fs::read_dir(OUT_DIR).ok()?.flatten() {
        let path = ent.path();
        if !path.is_dir() {
            continue;
        }
        let name = ent.file_name();
        let name = name.to_string_lossy();
        let Some(num) = name.strip_prefix(&prefix) else { continue };
        let Ok(n) = num.parse::<usize>() else { continue };
        if !path.join("optimiser_state").is_dir() {
            continue;
        }
        if best.as_ref().map_or(true, |(_, b)| n > *b) {
            best = Some((path.to_string_lossy().into_owned(), n));
        }
    }
    best
}

/// Standard bullet Stockfish-binpack position filter.
fn filter(entry: &TrainingDataEntry) -> bool {
    entry.ply >= 16
        && !entry.pos.is_checked(entry.pos.side_to_move())
        && entry.score.unsigned_abs() <= 10_000
        && entry.mv.mtype() == MoveType::Normal
        && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        usage();
    }

    // ---- resume point ----
    let (resume_dir, start_superbatch) = match latest_checkpoint() {
        Some((dir, n)) => {
            println!("resume: {dir} (completed superbatch {n}) -> continuing from {}", n + 1);
            (Some(dir), n + 1)
        }
        None => {
            println!("resume: no '{OUT_DIR}/{NET_ID}-<N>' checkpoint found -> starting fresh");
            (None, 1)
        }
    };
    if start_superbatch > END_SUPERBATCH {
        println!(
            "'{NET_ID}' is already trained to superbatch {END_SUPERBATCH}. Nothing to do. \
             (bump END_SUPERBATCH and rebuild to train longer.)"
        );
        return;
    }

    // ---- data files ----
    let mut files = collect_data_paths(&args);
    if ROTATE_DATA_EACH_SESSION && files.len() > 1 {
        let k = (start_superbatch - 1) % files.len();
        if k != 0 {
            files.rotate_left(k);
            println!("rotated {} data files left by {k} (per-session coverage spread)", files.len());
        }
    }
    let paths_ref: Vec<&str> = files.iter().map(String::as_str).collect();

    println!("--------------------------------------------------------------");
    println!("net id        : {NET_ID}   arch (768 -> {HIDDEN_SIZE}) x 2 -> 1 SCReLU");
    println!("quantisation  : QA={QA} QB={QB} eval_scale={EVAL_SCALE}");
    println!("superbatches  : {start_superbatch}..={END_SUPERBATCH}  ({BATCHES_PER_SUPERBATCH} x {BATCH_SIZE})");
    println!("lr            : {LR_START} -> {LR_FINAL} cosine");
    println!("wdl lambda    : {WDL_START} -> {WDL_END} linear");
    println!("save rate     : every {SAVE_RATE} superbatches -> {OUT_DIR}/");
    println!("data          : {} files, {DATA_THREADS} decode threads, {SHUFFLE_BUFFER_MB} MiB buffer", files.len());
    println!("--------------------------------------------------------------");

    // ---- build trainer ----
    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW)
        .inputs(Chess768)
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA),
            SavedFormat::id("l0b").round().quantise::<i16>(QA),
            SavedFormat::id("l1w").round().quantise::<i16>(QB),
            SavedFormat::id("l1b").round().quantise::<i16>(QA * QB),
        ])
        .loss_fn(|output, target| output.sigmoid().squared_error(target))
        .build(|builder, stm_inputs, ntm_inputs| {
            let l0 = builder.new_affine("l0", 768, HIDDEN_SIZE);
            let l1 = builder.new_affine("l1", 2 * HIDDEN_SIZE, 1);

            let stm_hidden = l0.forward(stm_inputs).screlu();
            let ntm_hidden = l0.forward(ntm_inputs).screlu();
            l1.forward(stm_hidden.concat(ntm_hidden))
        });

    if let Some(dir) = &resume_dir {
        // loads `<dir>/optimiser_state` (weights + Adam moments). Does NOT set
        // the start superbatch -- that's steps.start_superbatch below.
        trainer.load_from_checkpoint(dir);
    }

    let schedule = TrainingSchedule {
        net_id: NET_ID.to_string(),
        eval_scale: EVAL_SCALE as f32,
        steps: TrainingSteps {
            batch_size: BATCH_SIZE,
            batches_per_superbatch: BATCHES_PER_SUPERBATCH,
            start_superbatch,
            end_superbatch: END_SUPERBATCH,
        },
        // both schedulers key off the ABSOLUTE superbatch index, so they stay on
        // schedule across any number of resumes.
        wdl_scheduler: wdl::LinearWDL { start: WDL_START, end: WDL_END },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: LR_START,
            final_lr: LR_FINAL,
            final_superbatch: END_SUPERBATCH,
        },
        save_rate: SAVE_RATE,
    };

    let settings = LocalSettings {
        threads: CPU_THREADS,
        test_set: None, // validation not implemented in current bullet
        output_directory: OUT_DIR,
        batch_queue_size: 64,
    };

    let data_loader =
        SfBinpackLoader::new_concat_multiple(&paths_ref, SHUFFLE_BUFFER_MB, DATA_THREADS, filter);

    trainer.run(&schedule, &settings, &data_loader);

    println!(
        "done. deploy:  cp {OUT_DIR}/{NET_ID}-{END_SUPERBATCH}/quantised.bin  <engine>/nets/net.nnue"
    );
}
