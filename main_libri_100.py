import logging
import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Audio, Dataset, DatasetDict
from speakerbox import train, eval_model

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# Constants
LIBRI_ROOT = r"D:\libri100_WAV" 
OUTPUT_MODEL_NAME = "libri_prod_model_80_10_10"
MIN_FILES = 10  # LibriSpeech has many files per speaker, 10 is a safe minimum

def prepare_prod_dataset(root_path: str) -> DatasetDict:
    root = Path(root_path)
    data = []
    
    # LibriSpeech uses .flac mostly, but we check for .wav too
    audio_files = list(root.glob("**/*.flac")) + list(root.glob("**/*.wav"))
    log.info(f"Found {len(audio_files)} files. Processing...")

    if not audio_files:
        raise ValueError(f"No audio files found in {root_path}!")

    for filepath in audio_files:
        # Ignore macOS/System hidden files
        if filepath.name.startswith("._"):
            continue
        
        # Libri format: 121-121720-0003.flac -> parts[0]=Speaker, parts[1]=Chapter
        parts = filepath.stem.split("-")
        if len(parts) >= 3:
            data.append({
                "audio": str(filepath.resolve()), 
                "label": parts[0],             # Speaker ID
                "chapter_id": parts[1],        # Chapter ID
                "duration": 1.0 
            })
            
    df = pd.DataFrame(data)
    
    # Filter speakers with enough files
    counts = df.groupby("label")["audio"].count()
    valid_speakers = counts[counts >= MIN_FILES].index
    df = df[df.label.isin(valid_speakers)]
    log.info(f"Training on {len(valid_speakers)} speakers")

    # --- 80-10-10 SPLIT LOGIC ---
    # We use unique file paths to ensure a clean split
    unique_files = df["audio"].unique()

    # 1. Split off 80% for training
    train_files, temp_files = train_test_split(
        unique_files, test_size=0.20, random_state=42
    )

    # 2. Split the remaining 20% into two halves (10% Valid, 10% Test)
    valid_files, test_files = train_test_split(
        temp_files, test_size=0.50, random_state=42
    )

    dataset = DatasetDict({
        "train": Dataset.from_pandas(df[df["audio"].isin(train_files)], preserve_index=False),
        "valid": Dataset.from_pandas(df[df["audio"].isin(valid_files)], preserve_index=False),
        "test": Dataset.from_pandas(df[df["audio"].isin(test_files)], preserve_index=False)
    })

    # Resample to 16kHz (Standard for LibriSpeech and Wav2Vec2)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    dataset = dataset.class_encode_column("label")

    return dataset

def main():
    # Safety: Clean up old model folder to prevent checkpoint resume errors
    if Path(OUTPUT_MODEL_NAME).exists():
        log.info(f"Cleaning up existing directory: {OUTPUT_MODEL_NAME}")
        try:
            shutil.rmtree(OUTPUT_MODEL_NAME)
        except PermissionError:
            log.warning("Could not delete folder. Please close any open files in that directory.")

    dataset = prepare_prod_dataset(LIBRI_ROOT)
    
    print("STARTING LIBRISPEECH PRODUCTION TRAINING (80-10-10)")
    
    # 
    model_path = train(
        dataset=dataset,
        model_name=OUTPUT_MODEL_NAME,
        max_duration=2.0,
        trainer_arguments_kws={
            "gradient_accumulation_steps": 2,
            "per_device_train_batch_size": 4,
            "num_train_epochs": 3,
            "save_total_limit": 2,
            "fp16": False,                # Set to True ONLY if using NVIDIA CUDA
            "evaluation_strategy": "epoch",
            "dataloader_num_workers": 0,  # Windows stability fix
            "resume_from_checkpoint": False,
        }
    )
    
    metrics = eval_model(dataset["test"], str(model_path))
    print(f"Final Test Results: {metrics}")

if __name__ == "__main__":
    main()