#!/usr/bin/env python3

import logging
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

from speakerbox import train, eval_model

# -------------------------
# Logging
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger(__name__)

# -------------------------
# Config
# -------------------------
VCTK_ROOT = "/Users/napatkritasavarojpanich/Desktop/VCTK/VCTK-Corpus/wav16"
OUTPUT_MODEL_NAME = "vctk_prod_model_full"
MIN_FILES = 5

# (Optional) Reduce HF verbosity noise
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def prepare_prod_dataset(root_path: str) -> DatasetDict:
    root = Path(root_path)
    data = []

    audio_files = list(root.glob("**/*.flac")) + list(root.glob("**/*.wav"))
    log.info(f"Found {len(audio_files)} audio files. Processing...")

    for filepath in audio_files:
        parts = filepath.name.split("_")
        if len(parts) >= 2:
            data.append(
                {
                    "audio": str(filepath.resolve()),
                    "label": parts[0],
                    "conversation_id": parts[1],
                    "duration": 1.0,
                }
            )

    df = pd.DataFrame(data)
    if df.empty:
        raise RuntimeError("No valid audio files found (expected filename like <spk>_<conv>_...).")

    # keep speakers with >= MIN_FILES unique conversation ids
    counts = df.groupby("label")["conversation_id"].nunique()
    valid_speakers = counts[counts >= MIN_FILES].index
    df = df[df["label"].isin(valid_speakers)]
    log.info(f"Training on {len(valid_speakers)} speakers (filtered by MIN_FILES={MIN_FILES}).")
    log.info(f"Remaining rows: {len(df)}")

    unique_ids = df["conversation_id"].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

    train_df = df[df["conversation_id"].isin(train_ids)]
    test_df = df[df["conversation_id"].isin(test_ids)]

    dataset = DatasetDict(
        {
            "train": Dataset.from_pandas(train_df, preserve_index=False),
            "test": Dataset.from_pandas(test_df, preserve_index=False),
        }
    )

    dataset = dataset.class_encode_column("label")
    return dataset


def main():
    dataset = prepare_prod_dataset(VCTK_ROOT)

    log.info("STARTING PRODUCTION TRAINING")

    # IMPORTANT:
    # - If your train() uses a custom collator / function defined inside train(),
    #   macOS multiprocessing will crash with "Can't pickle local object".
    #   So keep dataloader_num_workers=0 unless you moved collator to module scope.
    trainer_arguments_kws = {
        "gradient_accumulation_steps": 2,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 1,
        "num_train_epochs": 3,

        "fp16": False,
        "learning_rate": 1e-5,
        "max_grad_norm": 1.0,

        "logging_strategy": "steps",
        "logging_steps": 100,

        # if your speakerbox uses "eval_strategy" (newer key), keep it as is:
        "eval_strategy": "steps",
        "eval_steps": 4000,
        "eval_accumulation_steps": 1,

        "save_strategy": "steps",
        "save_steps": 4000,
        "save_total_limit": 2,

        # SAFE defaults for macOS (avoid pickle crash)
        "dataloader_num_workers": 0,
        "dataloader_persistent_workers": False,
        "dataloader_pin_memory": False,

        "gradient_checkpointing": False,
    }

    model_path = train(
        dataset=dataset,
        model_name=OUTPUT_MODEL_NAME,
        max_duration=2.0,
        trainer_arguments_kws=trainer_arguments_kws,
    )

    log.info(f"Training done. Model saved at: {model_path}")

    # Final evaluation on full test set (this can take a while)
    metrics = eval_model(dataset["test"], str(model_path))
    log.info(f"Results: {metrics}")
    print(f"Results: {metrics}")


if __name__ == "__main__":
    main()
