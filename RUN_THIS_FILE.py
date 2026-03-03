import argparse
import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Audio, Dataset, DatasetDict
from torch.cuda import seed
from transformers import data
from speakerbox.tests import data
from speakerbox.main import train, eval_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Dataset builder

def prepare_dataset(root_path: str, min_speaker_files: int, seed: int, max_samples: int = None) -> DatasetDict:    
    """
    Scans root_path recursively for .wav / .flac files.
    Derives the speaker label from the filename prefix before the first '_'.
    Returns an 80 / 10 / 10 train / valid / test DatasetDict with a FIXED seed.
    """
    root = Path(root_path)
    data = []

    audio_files = list(root.glob("**/*.flac")) + list(root.glob("**/*.wav"))
    log.info(f"Found {len(audio_files)} audio files.")

    for filepath in audio_files:
        parts = filepath.name.split("_")
        if len(parts) >= 2:
            data.append({
                "audio":           str(filepath.resolve()),
                "label":           parts[0],
                "conversation_id": parts[1],
            })

    df = pd.DataFrame(data).sample(frac=1, random_state=seed).reset_index(drop=True)

    if max_samples and max_samples > 0:
        log.info(f"Limiting dataset to {max_samples} samples.")
        df = df.head(max_samples)
    
    counts = df.groupby("label")["conversation_id"].nunique()
    valid_speakers = counts[counts >= min_speaker_files].index
    df = df[df["label"].isin(valid_speakers)]
    log.info(f"Training on {len(valid_speakers)} speakers after filtering.")

    unique_files = df["audio"].unique()
    train_files, temp_files = train_test_split(unique_files, test_size=0.20, random_state=seed)
    valid_files, test_files = train_test_split(temp_files,   test_size=0.50, random_state=seed)

    ds = DatasetDict({
        "train": Dataset.from_pandas(df[df["audio"].isin(train_files)], preserve_index=False),
        "valid": Dataset.from_pandas(df[df["audio"].isin(valid_files)], preserve_index=False),
        "test":  Dataset.from_pandas(df[df["audio"].isin(test_files)],  preserve_index=False),
    })

    ds = ds.cast_column("audio", Audio(decode=False))
    ds = ds.class_encode_column("label")
    return ds

# Sub-command handlers

def cmd_train(args):
    dataset = prepare_dataset(args.dataset, args.min_speaker_files, args.seed, args.max_samples)
    trainer_args = {
        "learning_rate":                args.lr,
        "num_train_epochs":             args.epochs,
        "per_device_train_batch_size":  args.batch,
        "gradient_accumulation_steps":  args.accum,
        "fp16":                         args.fp16,
        "evaluation_strategy":          "no",
        "save_strategy":                "steps",
        "save_steps":                   args.save_steps,
        "save_total_limit":             3,
        "dataloader_num_workers":       0,   # keep 0 on Windows
        "remove_unused_columns":        False,
        "logging_steps":                10,
        "report_to":                    "none",
    }

    model_path = train(
        dataset=dataset,
        model_name=args.output,
        model_base=args.model_base,
        max_duration=args.max_duration,
        trainer_arguments_kws=trainer_args,
        audio_backend=args.audio_backend,
        metadata_cache_path=args.metadata_cache,
        eval_mode=args.eval_mode,
        seed=args.seed,
    )
    log.info(f"Model saved -> {model_path}")
    return dataset


def cmd_eval(args, dataset=None):
    if dataset is None:
        dataset = prepare_dataset(args.dataset, args.min_speaker_files, args.seed)

    dataset["test"] = dataset["test"].cast_column("audio", Audio(decode=False))

    metrics = eval_model(
        validation_dataset=dataset["test"],
        model_name=args.eval_model or args.output,
        eval_mode=args.eval_mode,
        audio_backend=args.audio_backend,
    )
    log.info(
        f"Eval results -- "
        f"Accuracy: {metrics[0]:.4f} | Precision: {metrics[1]:.4f} | "
        f"Recall: {metrics[2]:.4f} | Loss: {metrics[3]:.4f}"
    )

# CLI definition

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Speakerbox -- train and/or evaluate a speaker-ID model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    sub = parser.add_subparsers(dest="command", required=True)
    p_train      = sub.add_parser("train",      help="Train only")
    p_eval       = sub.add_parser("eval",        help="Evaluate only")
    p_train_eval = sub.add_parser("train_eval",  help="Train then evaluate")

    # ---- shared arguments (added to every sub-command) ----
    for p in (p_train, p_eval, p_train_eval):
        p.add_argument(
            "--dataset", required=True, metavar="PATH",
            help="Root directory containing speaker audio files.",
        )
        p.add_argument(
            "--output", default="exps/model", metavar="PATH",
            help="Output directory for the trained model. (default: exps/model)",
        )
        p.add_argument(
            "--audio-backend", choices=["soundfile", "librosa"], default="soundfile",
            help=(
                "soundfile: fast, assumes files are already 16 kHz. "
                "librosa: slower, auto-resamples to target SR. "
                "(default: soundfile)"
            ),
        )
        p.add_argument(
            "--min-speaker-files", type=int, default=1, metavar="N",
            help="Minimum unique files per speaker to be included. (default: 1)",
        )
        p.add_argument(
        "--max-samples", type=int, default=None, metavar="N",
        help="Total number of audio samples to use (useful for quick tests).",
        )
        p.add_argument(
            "--eval-mode", choices=["softmax", "pipeline"], default="softmax",
            help=(
                "softmax:  batched forward pass + softmax (mac/MPS/CPU friendly). "
                "pipeline: HF pipeline, one sample at a time (windows-style). "
                "(default: softmax)"
            ),
        )

    # ---- train-specific arguments ----
    for p in (p_train, p_train_eval):
        p.add_argument(
            "--model-base", default="superb/wav2vec2-base-superb-sid", metavar="HF_ID",
            help="HuggingFace model ID to fine-tune from. "
                 "(default: superb/wav2vec2-base-superb-sid)",
        )
        p.add_argument(
            "--max-duration", type=float, default=3.0, metavar="SEC",
            help="Maximum audio clip length in seconds; clips are truncated/padded. "
                 "(default: 3.0)",
        )
        p.add_argument(
            "--epochs", type=int, default=10, metavar="N",
            help="Number of training epochs. (default: 10)",
        )
        p.add_argument(
            "--batch", type=int, default=4, metavar="N",
            help="Per-device train batch size. (default: 4)",
        )
        p.add_argument(
            "--accum", type=int, default=4, metavar="N",
            help="Gradient accumulation steps; effective batch = batch x accum. (default: 4)",
        )
        p.add_argument(
            "--lr", type=float, default=3e-5, metavar="LR",
            help="Learning rate. (default: 3e-5)",
        )
        p.add_argument(
            "--fp16", action="store_true",
            help="Enable mixed-precision training (requires NVIDIA GPU).",
        )
        p.add_argument(
            "--save-steps", type=int, default=200, metavar="N",
            help="Save a checkpoint every N steps. (default: 200)",
        )
        p.add_argument(
            "--metadata-cache", default=None, metavar="PATH",
            help=(
                "Path to cache the Audio-cast dataset (speeds up reruns). "
                "Omit to use default location inside --output. "
                "Pass empty string '' to disable caching."
            ),
        )
        p.add_argument(
            "--seed", type=int, default=42, metavar="N",
            help="Global random seed. (default: 42)",
        )

    # ---- eval-specific arguments ----
    for p in (p_eval, p_train_eval):
        p.add_argument(
            "--eval-model", default=None, metavar="PATH",
            help="Path to the model to evaluate. Defaults to --output if not specified.",
        )
        
        
    # ---- mac shortcut ----
    p_mac = sub.add_parser(
        "mac",
        help="Preset for Apple Silicon (MPS): soundfile backend, softmax eval, fp16 off.",
    )
    p_mac.add_argument("--dataset",  required=True, metavar="PATH",
                       help="Root directory of audio files.")
    p_mac.add_argument("--output",   default="exps/model", metavar="PATH",
                       help="Output directory. (default: exps/model)")
    p_mac.add_argument("--epochs",   type=int,   default=10,   metavar="N")
    p_mac.add_argument("--batch",    type=int,   default=4,    metavar="N")
    p_mac.add_argument("--accum",    type=int,   default=4,    metavar="N")
    p_mac.add_argument("--lr",       type=float, default=3e-5, metavar="LR")
    p_mac.add_argument("--max-duration", type=float, default=3.0, metavar="SEC")
    p_mac.add_argument("--save-steps",   type=int,   default=200,  metavar="N")
    p_mac.add_argument("--min-speaker-files", type=int, default=1, metavar="N")
    p_mac.add_argument("--metadata-cache", default=None, metavar="PATH")
    p_mac.add_argument("--seed",     type=int,   default=42,   metavar="N")
    p_mac.add_argument("--eval-model", default=None, metavar="PATH",
                       help="Model to eval after training. Defaults to --output.")
    p_mac.add_argument("--resume",    action="store_true",  default=True)
    p_mac.add_argument("--no-resume", dest="resume", action="store_false")
    p_mac.add_argument( "--max-samples", type=int, default=None, metavar="N")

    # ---- windows shortcut ----
    p_win = sub.add_parser(
        "windows",
        help="Preset for Windows/NVIDIA (CUDA): librosa backend, pipeline eval, fp16 on.",
    )
    p_win.add_argument("--dataset",  required=True, metavar="PATH",
                       help="Root directory of audio files.")
    p_win.add_argument("--output",   default="exps/model", metavar="PATH",
                       help="Output directory. (default: exps/model)")
    p_win.add_argument("--epochs",   type=int,   default=10,   metavar="N")
    p_win.add_argument("--batch",    type=int,   default=4,    metavar="N")
    p_win.add_argument("--accum",    type=int,   default=4,    metavar="N")
    p_win.add_argument("--lr",       type=float, default=3e-5, metavar="LR")
    p_win.add_argument("--max-duration", type=float, default=3.0, metavar="SEC")
    p_win.add_argument("--save-steps",   type=int,   default=200,  metavar="N")
    p_win.add_argument("--min-speaker-files", type=int, default=1, metavar="N")
    p_win.add_argument("--metadata-cache", default=None, metavar="PATH")
    p_win.add_argument("--seed",     type=int,   default=42,   metavar="N")
    p_win.add_argument("--eval-model", default=None, metavar="PATH",
                       help="Model to eval after training. Defaults to --output.")
    p_win.add_argument("--resume",    action="store_true",  default=True)
    p_win.add_argument("--no-resume", dest="resume", action="store_false")
    p_win.add_argument( "--max-samples", type=int, default=None, metavar="N")

    return parser



# Entry point


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "train_eval":
        dataset = cmd_train(args)
        cmd_eval(args, dataset=dataset)
    elif args.command == "mac":
        # Apple Silicon preset: MPS device, soundfile, softmax eval, fp16 off
        args.model_base   = "superb/wav2vec2-base-superb-sid"
        args.audio_backend = "soundfile"
        args.eval_mode    = "softmax"
        args.fp16         = False
        dataset = cmd_train(args)
        args.eval_model   = args.eval_model or args.output
        cmd_eval(args, dataset=dataset)

    elif args.command == "windows":
        # Windows/NVIDIA preset: CUDA, librosa, pipeline eval, fp16 on
        args.model_base   = "superb/wav2vec2-base-superb-sid"
        args.audio_backend = "librosa"
        args.eval_mode    = "pipeline"
        args.fp16         = True
        dataset = cmd_train(args)
        args.eval_model   = args.eval_model or args.output
        cmd_eval(args, dataset=dataset)


if __name__ == "__main__":
    main()