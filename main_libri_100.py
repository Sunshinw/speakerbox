import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from speakerbox import train, eval_model

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Constants
LIBRI_ROOT = "path/to/LibriSpeech"  # Update this to your local path
OUTPUT_MODEL_NAME = "librispeech_speakerbox_model"
MIN_FILES_PER_SPEAKER = 10 

def prepare_librispeech_dataset(root_path: str) -> DatasetDict:
    root = Path(root_path)
    data = []
    
    # LibriSpeech usually uses .flac
    audio_files = list(root.glob("**/*.flac")) + list(root.glob("**/*.wav"))
    log.info(f"Found {len(audio_files)} files in {root_path}")

    for filepath in audio_files:
        # Ignore macOS metadata files
        if filepath.name.startswith("._"):
            continue

        # LibriSpeech format: speakerID-chapterID-utteranceID.flac
        parts = filepath.stem.split("-")
        if len(parts) >= 3:
            data.append({
                "audio": str(filepath.resolve()), 
                "label": parts[0],             # Speaker ID
                "chapter_id": parts[1],        # Chapter ID for grouped splitting
            })
            
    df = pd.DataFrame(data)
    
    # Filter for speakers with enough data
    counts = df.groupby("label")["audio"].count()
    valid_speakers = counts[counts >= MIN_FILES_PER_SPEAKER].index
    df = df[df.label.isin(valid_speakers)]
    log.info(f"Training on {len(valid_speakers)} speakers with 100% of available data.")

    # Split by Chapter ID to ensure the model generalizes to new recordings of the same speaker
    unique_chapters = df.chapter_id.unique()
    train_chapters, test_chapters = train_test_split(unique_chapters, test_size=0.2, random_state=42)

    train_df = df[df.chapter_id.isin(train_chapters)]
    test_df = df[df.chapter_id.isin(test_chapters)]

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df.drop(columns=["chapter_id"]), preserve_index=False),
        "test": Dataset.from_pandas(test_df.drop(columns=["chapter_id"]), preserve_index=False)
    })

    # Convert labels to class indices
    dataset = dataset.class_encode_column("label")

    return dataset

def main():
    # 1. Prepare Data
    dataset = prepare_librispeech_dataset(LIBRI_ROOT)
    
    print(f"STARTING TRAINING: {OUTPUT_MODEL_NAME}")
    
    # 2. Train using Speakerbox
    # This handles feature extraction, model init, and the training loop
    model_path = train(
        dataset=dataset,
        model_name=OUTPUT_MODEL_NAME,
        max_duration=2.0,
        trainer_arguments_kws={
            "gradient_accumulation_steps": 2,
            "per_device_train_batch_size": 8,
            "num_train_epochs": 3,
            "save_total_limit": 1,
            "learning_rate": 3e-5,
            "fp16": False, # Set to True if using a CUDA GPU; False for Mac MPS
            "logging_steps": 10,
            "evaluation_strategy": "epoch"
        }
    )
    
    # 3. Evaluation
    log.info("Evaluating model...")
    metrics = eval_model(dataset["test"], str(model_path))
    print(f"Final Metrics: {metrics}")

if __name__ == "__main__":
    main()