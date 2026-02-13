import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Audio, Dataset, DatasetDict
from speakerbox import eval_model

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

VCTK_ROOT = "/Users/pakap/Documents/Senior/Code/Dataset/libri500_WAV"
EPOCH1_MODEL_PATH = "exps/libri500/final_speakerbox_epoch_1"
MIN_FILES = 1


def prepare_prod_dataset(root_path: str) -> DatasetDict:
    root = Path(root_path)
    data = []

    audio_files = list(root.glob("**/*.flac")) + list(root.glob("**/*.wav"))
    log.info(f"Found {len(audio_files)} files. Processing...")

    for filepath in audio_files:
        parts = filepath.name.split("_")
        if len(parts) >= 2:
            data.append({
                "audio": str(filepath.resolve()),
                "label": parts[0],
                "conversation_id": parts[1],
                "duration": 1.0,
            })

    df = pd.DataFrame(data)

    counts = df.groupby("label")["conversation_id"].nunique()
    valid_speakers = counts[counts >= MIN_FILES].index
    df = df[df.label.isin(valid_speakers)]
    log.info(f"Eval on {len(valid_speakers)} speakers")

    unique_files = df["audio"].unique()

    train_files, temp_files = train_test_split(
        unique_files, test_size=0.20, random_state=42
    )
    valid_files, test_files = train_test_split(
        temp_files, test_size=0.50, random_state=42
    )

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df[df["audio"].isin(train_files)], preserve_index=False),
        "valid": Dataset.from_pandas(df[df["audio"].isin(valid_files)], preserve_index=False),
        "test": Dataset.from_pandas(df[df["audio"].isin(test_files)], preserve_index=False),
    })

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000, decode=False))

    dataset = dataset.class_encode_column("label")

    return dataset


def main():
    dataset = prepare_prod_dataset(VCTK_ROOT)

    log.info("Running EVAL ONLY for epoch 1 model...")
    metrics = eval_model(dataset["valid"], model_name=EPOCH1_MODEL_PATH)

    print("\n✅ DONE")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
