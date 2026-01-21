import logging
import os
import torch
import numpy as np
import pandas as pd
import librosa
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_metric
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)

# 1. Setup Logging & Constants
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

LIBRI_ROOT = "*" 
OUTPUT_MODEL_DIR = "libri_fast_model"

BATCH_SIZE = 8
NUM_EPOCHS = 3
LEARNING_RATE = 3e-5
MAX_DURATION = 2.0
SUBSET_PERCENTAGE = 0.1

# 2. Data Preparation
def prepare_librispeech_subset(root_path: str, subset_percentage: float = 0.1) -> pd.DataFrame:
    root = Path(root_path)
    log.info(f"Scanning LibriSpeech files in: {root}")
    
    data = []
    # Mac files are usually .wav or .flac
    audio_files = list(root.glob("**/*.wav")) + list(root.glob("**/*.flac"))

    if not audio_files:
        raise ValueError(f"No audio files found in {root_path}!")

    for filepath in audio_files:
        
        if filepath.name.startswith("."):
            continue

        parts = filepath.stem.split("-") 
        if len(parts) >= 3:
            data.append({
                "audio": str(filepath.resolve()), 
                "label": parts[0],
                "chapter_id": parts[1]
            })
            
    df = pd.DataFrame(data)
    counts = df.groupby("label")["audio"].count()
    valid_speakers = counts[counts >= 10].index
    df = df[df.label.isin(valid_speakers)]
    
    log.info(f"Subsampling to {subset_percentage*100}%...")
    df_subset = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=subset_percentage, random_state=42)
    )
    return df_subset

def create_split(df: pd.DataFrame) -> DatasetDict:
    train_ids, test_valid_ids = train_test_split(df.chapter_id.unique(), test_size=0.2, random_state=42)
    test_ids, valid_ids = train_test_split(test_valid_ids, test_size=0.5, random_state=42)

    train_ds = df[df.chapter_id.isin(train_ids)]
    test_ds = df[df.chapter_id.isin(test_ids)]
    valid_ds = df[df.chapter_id.isin(valid_ids)]
    
    return DatasetDict({
        "train": Dataset.from_pandas(train_ds.drop(columns=["chapter_id"]), preserve_index=False),
        "test": Dataset.from_pandas(test_ds.drop(columns=["chapter_id"]), preserve_index=False),
        "valid": Dataset.from_pandas(valid_ds.drop(columns=["chapter_id"]), preserve_index=False),
    })

# 3. Main Training Logic
def main():
    # 🔥 Hardware Check for Mac
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple Silicon (MPS) acceleration")
    else:
        device = torch.device("cpu")
        log.info("MPS not found, using CPU")

    df = prepare_librispeech_subset(LIBRI_ROOT, SUBSET_PERCENTAGE)
    dataset = create_split(df)
    
    dataset = dataset.class_encode_column("label")
    label2id = {l: i for i, l in enumerate(dataset["train"].features["label"].names)}
    id2label = {str(i): l for l, i in label2id.items()}
    num_labels = len(label2id)

    base_model = "superb/wav2vec2-base-superb-sid"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model)

    def preprocess_function(audio_list, label_list):
        audio_arrays = []
        for path in audio_list:
            actual_path = path["path"] if isinstance(path, dict) else path
            try:
                # Double check for macOS ghost files
                if os.path.basename(actual_path).startswith("._"):
                    raise ValueError("Skipping metadata")
                
                speech, _ = librosa.load(actual_path, sr=16000)
                audio_arrays.append(speech)
            except Exception:
                audio_arrays.append(np.zeros(int(16000 * MAX_DURATION)))

        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            max_length=int(16000 * MAX_DURATION),
            truncation=True,
            padding="max_length",
            do_normalize=True,
        )
        inputs["labels"] = label_list 
        return inputs

    log.info("Pre-processing audio...")
    encoded_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=10, 
        input_columns=["audio", "label"], 
        remove_columns=["audio", "label"],
        load_from_cache_file=False 
    )
    
    encoded_dataset.set_format(type="torch", columns=["input_values", "labels"])

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    ).to(device) # Move model to MPS

    metric = load_metric("accuracy", trust_remote_code=True)
    def compute_metrics(eval_pred: EvalPrediction):
        logits = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    # 🔥 Training Config for Mac
    args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=4, 
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        fp16=False, 
        bf16=False, 
        eval_accumulation_steps=1,
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["valid"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_MODEL_DIR, "final_model"))

if __name__ == "__main__":
    main()