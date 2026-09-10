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

/// Training length. Instead of a fixed superbatch count, the run length is
/// derived at startup from the input size so it's ~EPOCHS passes over the data:
///
///   end_superbatch = round(EPOCHS * total_positions * FILTER_KEEP_FRAC
///                          / (BATCHES_PER_SUPERBATCH * BATCH_SIZE))
///
/// `total_positions` is looked up per input file. Precedence, per file:
///   1. `<file>.binpack.count` sidecar (plain integer; `_` / `,` / ws ignored)
///   2. the `POSITION_COUNTS` table below, keyed by file *basename*
/// A file matching neither is fatal. Env `EREBUS_END_SUPERBATCH=N` bypasses the
/// whole calc.
const EPOCHS: f64 = 1.0;

/// Known raw position counts, keyed by binpack basename (no directory), from
/// `binpack_counter`. Order doesn't matter. Sum = 218_849_949_380 over 41 files.
const POSITION_COUNTS: &[(&str, u64)] = &[
    ("test60-2021-11-nov-12tb7p.min-v2.relabel-BT4-tf13tune.binpack", 1_452_424_355),
    ("test60-2021-12-dec-12tb7p.min-v2.relabel-BT4-tf13tune.binpack", 1_363_206_227),
    ("test77-2021-12-dec-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 6_286_624_013),
    ("test78-2022-01-to-05-jantomay-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 7_419_909_666),
    ("test78-2022-06-to-09-juntosep-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 3_574_170_531),
    ("test79-2022-04-apr-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 2_930_087_205),
    ("test79-2022-05-may-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 2_284_179_270),
    ("test80-2022-06-jun-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_488_679_928),
    ("test80-2022-07-jul-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_835_573_847),
    ("test80-2022-08-aug-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 3_801_599_910),
    ("test80-2022-09-sep-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_171_904_814),
    ("test80-2022-10-oct-16tb7p.v6-dd.relabel-BT4-tf13tune.part_0.binpack", 2_030_804_185),
    ("test80-2022-10-oct-16tb7p.v6-dd.relabel-BT4-tf13tune.part_1.binpack", 2_030_725_513),
    ("test80-2022-11-nov-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 4_608_318_891),
    ("test80-2023-01-jan-16tb7p.v6-sk20.min.relabel-BT4-tf13tune.binpack", 4_707_093_556),
    ("test80-2023-02-feb-16tb7p.v6-dd.min.relabel-BT4-tf13tune.binpack", 3_626_845_354),
    ("test80-2023-03-mar-2tb7p.v6-sk16.min.relabel-BT4-tf13tune.binpack", 5_520_899_664),
    ("test80-2023-04-apr-2tb7p.v6-sk16.min.relabel-BT4-tf13tune.binpack", 5_653_619_110),
    ("test80-2023-05-may-2tb7p.v6.min.relabel-BT4-tf13tune.binpack", 5_600_480_538),
    ("test80-2023-06-jun-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 6_756_356_195),
    ("test80-2023-07-jul-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 6_212_977_488),
    ("test80-2023-08-aug-2tb7p.v6.min.relabel-BT4-tf13tune.binpack", 2_693_519_136),
    ("test80-2023-09-sep-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 3_257_611_143),
    ("test80-2023-10-oct-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 3_012_783_968),
    ("test80-2023-11-nov-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 2_724_311_169),
    ("test80-2023-12-dec-2tb7p.min-v2.v6.relabel-BT4-tf13tune.binpack", 3_016_184_922),
    ("leela96-filt-v2.min.split_0.relabel-BT4-tf13tune.binpack", 5_681_356_602),
    ("leela96-filt-v2.min.split_1.relabel-BT4-tf13tune.binpack", 5_679_303_898),
    ("leela96-filt-v2.min.split_2.relabel-BT4-tf13tune.binpack", 5_680_096_474),
    ("leela96-filt-v2.min.split_3.relabel-BT4-tf13tune.binpack", 5_681_363_129),
    ("leela96-filt-v2.min.split_4.relabel-BT4-tf13tune.binpack", 5_680_919_670),
    ("T60T70wIsRightFarseerT60T74T75T76.split_0.relabel-BT4-tf13tune.binpack", 9_133_725_682),
    ("T60T70wIsRightFarseerT60T74T75T76.split_1.relabel-BT4-tf13tune.binpack", 9_150_872_906),
    ("T60T70wIsRightFarseerT60T74T75T76.split_2.relabel-BT4-tf13tune.binpack", 9_136_446_438),
    ("T60T70wIsRightFarseerT60T74T75T76.split_3.relabel-BT4-tf13tune.binpack", 9_128_728_654),
    ("T60T70wIsRightFarseerT60T74T75T76.split_4.relabel-BT4-tf13tune.binpack", 9_160_229_495),
    ("dfrc_n5000.relabel-BT4-tf13tune.binpack", 12_353_351_142),
    ("fishpack32.relabel-BT4-tf13tune.binpack", 2_555_358_353),
    ("multinet_pv-2_diff-100_nodes-5000.relabel-BT4-tf13tune.binpack", 9_485_503_089),
    ("nodes5000pv2_UHO.relabel-BT4-tf13tune.binpack", 13_937_427_120),
    ("wrongIsRight_nodes5000pv2.relabel-BT4-tf13tune.binpack", 2_344_376_130),
];
/// Fraction of raw positions expected to survive `filter()`. 1.0 treats the raw
/// count as the training-position budget (the loader may wrap slightly at the
/// end); lower it (~0.6) if bullet logs that it looped the data before finishing.
const FILTER_KEEP_FRAC: f64 = 1.0;
/// Safety cap on the derived length -- a bad count can't launch a runaway run.
/// One epoch of all 41 binpacks (~219B pos) is ~2188 superbatches, so this
/// allows roughly 2 epochs of the full set before clamping (and warning).
const MAX_SUPERBATCH: usize = 5000;
/// Positions per batch.
const BATCH_SIZE: usize = 16_384;
/// Batches per superbatch. 6104 * 16384 ~= 100M positions / superbatch.
const BATCHES_PER_SUPERBATCH: usize = 6104;
/// Save a checkpoint every this many superbatches (also always saves the last).
const SAVE_RATE: usize = 10;

/// Learning rate: cosine decay from LR_START to LR_FINAL over the whole run.
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

/// Raw position count for one input file: `<path>.count` sidecar if present,
/// else the `POSITION_COUNTS` table keyed by basename. Neither => fatal.
fn count_for(path: &str) -> u64 {
    let side = format!("{path}.count");
    if let Ok(raw) = std::fs::read_to_string(&side) {
        let digits: String = raw.chars().filter(char::is_ascii_digit).collect();
        return digits.parse().unwrap_or_else(|e| die(format!("{side}: no valid integer ({e})")));
    }

    let base = std::path::Path::new(path)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());

    match POSITION_COUNTS.iter().find(|(name, _)| *name == base) {
        Some((_, n)) => *n,
        None => die(format!(
            "no position count for '{base}': add it to POSITION_COUNTS in src/main.rs, \
             or drop a '{side}' file next to it, or set EREBUS_END_SUPERBATCH=N"
        )),
    }
}

/// Sum of raw position counts across all input files.
fn count_positions(files: &[String]) -> u64 {
    let mut total: u64 = 0;
    for f in files {
        let n = count_for(f);
        if n == 0 {
            die(format!("{f}: position count is 0"));
        }
        total += n;
    }
    total
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

    // ---- training length: env override, else derive ~EPOCHS passes from data size ----
    let pos_per_superbatch = BATCHES_PER_SUPERBATCH * BATCH_SIZE;
    let end_superbatch: usize = match std::env::var("EREBUS_END_SUPERBATCH") {
        Ok(v) => {
            let n = v
                .trim()
                .parse::<usize>()
                .unwrap_or_else(|e| die(format!("EREBUS_END_SUPERBATCH not a number: {e}")));
            println!("END_SUPERBATCH = {n} (from EREBUS_END_SUPERBATCH)");
            n.max(1)
        }
        Err(_) => {
            let total = count_positions(&files);
            let want = (EPOCHS * total as f64 * FILTER_KEEP_FRAC / pos_per_superbatch as f64).round()
                as usize;
            let n = want.clamp(1, MAX_SUPERBATCH);
            println!(
                "data positions: {total} (sidecar / POSITION_COUNTS)  ->  END_SUPERBATCH = {n}  \
                 (EPOCHS={EPOCHS}, filter_keep={FILTER_KEEP_FRAC}, {pos_per_superbatch} pos/superbatch)"
            );
            if n != want {
                println!("  (clamped from {want} by MAX_SUPERBATCH={MAX_SUPERBATCH})");
            }
            n
        }
    };
    if start_superbatch > end_superbatch {
        println!(
            "'{NET_ID}' is already trained to superbatch {end_superbatch}. Nothing to do. \
             (raise EPOCHS or set EREBUS_END_SUPERBATCH, then rerun to train longer.)"
        );
        return;
    }

    println!("--------------------------------------------------------------");
    println!("net id        : {NET_ID}   arch (768 -> {HIDDEN_SIZE}) x 2 -> 1 SCReLU");
    println!("quantisation  : QA={QA} QB={QB} eval_scale={EVAL_SCALE}");
    println!("superbatches  : {start_superbatch}..={end_superbatch}  ({BATCHES_PER_SUPERBATCH} x {BATCH_SIZE})");
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
            end_superbatch,
        },
        // both schedulers key off the ABSOLUTE superbatch index, so they stay on
        // schedule across any number of resumes.
        wdl_scheduler: wdl::LinearWDL { start: WDL_START, end: WDL_END },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: LR_START,
            final_lr: LR_FINAL,
            final_superbatch: end_superbatch,
        },
        // never coarser than the run itself (bullet always saves the last too).
        save_rate: SAVE_RATE.min(end_superbatch).max(1),
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
        "done. deploy:  cp {OUT_DIR}/{NET_ID}-{end_superbatch}/quantised.bin  <engine>/nets/net.nnue"
    );
}
