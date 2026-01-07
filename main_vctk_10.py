import logging
import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, Audio, load_metric
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EvalPrediction,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# VCTK_ROOT = r"C:\Users\sunshine\Desktop\DS_10283_3443\VCTK-Corpus-0.92\wav48_silence_trimmed"
VCTK_ROOT = ""
OUTPUT_MODEL_DIR = "vctk_fast_model"

BATCH_SIZE = 8
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5
MAX_DURATION = 2.0
SUBSET_PERCENTAGE = 0.1

def prepare_vctk_subset(root_path: str) -> pd.DataFrame:
    root = Path(root_path)
    log.info(f"Scanning VCTK files in: {root}")
    
    data = []
    audio_files = list(root.glob("**/*.flac"))
    
    if not audio_files:
        raise ValueError("No .flac files found!")

    for filepath in audio_files:
        parts = filepath.name.split("_")
        if len(parts) >= 2:
            data.append({
                "audio": str(filepath.resolve()), 
                "label": parts[0],
                "conversation_id": parts[1],
            })
            
    df = pd.DataFrame(data)
    
    counts = df.groupby("label")["conversation_id"].nunique()
    valid_speakers = counts[counts >= 10].index
    df = df[df.label.isin(valid_speakers)]
    
    log.info(f"Subsampling to {SUBSET_PERCENTAGE*100}% of data...")
    
    df_subset = df.groupby("label", group_keys=False).apply(
        lambda x: x.sample(frac=SUBSET_PERCENTAGE, random_state=42)
    )
    
    log.info(f"Subset Ready: {len(df_subset)} samples (was {len(df)})")
    log.info(f"Speakers: {df_subset.label.nunique()}")
    
    return df_subset

def create_split(df: pd.DataFrame) -> DatasetDict:
    train_ids, test_valid_ids = train_test_split(df.conversation_id.unique(), test_size=0.2, random_state=42)
    test_ids, valid_ids = train_test_split(test_valid_ids, test_size=0.5, random_state=42)

    train_ds = df[df.conversation_id.isin(train_ids)]
    test_ds = df[df.conversation_id.isin(test_ids)]
    valid_ds = df[df.conversation_id.isin(valid_ids)]

    return DatasetDict({
        "train": Dataset.from_pandas(train_ds.drop(columns=["conversation_id"]), preserve_index=False),
        "test": Dataset.from_pandas(test_ds.drop(columns=["conversation_id"]), preserve_index=False),
        "valid": Dataset.from_pandas(valid_ds.drop(columns=["conversation_id"]), preserve_index=False),
    })

def main():
    df = prepare_vctk_subset(VCTK_ROOT)
    dataset = create_split(df)
    
    dataset = dataset.class_encode_column("label")
    label2id = {l: i for i, l in enumerate(dataset["train"].features["label"].names)}
    id2label = {str(i): l for l, i in label2id.items()}
    num_labels = len(label2id)

    base_model = "superb/wav2vec2-base-superb-sid"
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    def preprocess_function(examples):
        audio_arrays = [x["array"] for x in examples["audio"]]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=16000,
            max_length=int(16000 * MAX_DURATION),
            truncation=True,
            padding="max_length",
            do_normalize=True,
        )
        return inputs

    log.info("Pre-processing audio...")
    encoded_dataset = dataset.map(
        preprocess_function, 
        batched=True, 
        batch_size=100,
        remove_columns=["audio"]
    )
    
    encoded_dataset.set_format(type="torch", columns=["input_values", "label"])
    encoded_dataset = encoded_dataset.rename_column("label", "labels")

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )

    metric = load_metric("accuracy", trust_remote_code=True)
    def compute_metrics(eval_pred: EvalPrediction):
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
            
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    use_fp16 = torch.cuda.is_available()
    
    args = TrainingArguments(
        output_dir=OUTPUT_MODEL_DIR,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=0.1,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["valid"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
    )

    print("\n" + "="*50)
    print(f"STARTING FAST TRAINING (10% Data Subset)")
    print(f"Speakers: {num_labels}")
    print(f"Train Samples: {len(encoded_dataset['train'])}")
    print("="*50)
    
    trainer.train()

    print("Evaluating...")
    metrics = trainer.evaluate(encoded_dataset["test"])
    print(metrics)
    
    trainer.save_model(os.path.join(OUTPUT_MODEL_DIR, "final_model"))

if __name__ == "__main__":
    main()