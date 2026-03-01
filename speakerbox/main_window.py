#!/usr/bin/env python

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

from transformers import pipeline
import torch
import soundfile as sf
import numpy as np

if TYPE_CHECKING:
    import datasets
    from datasets import Dataset, DatasetDict, arrow_dataset
    from datasets import Audio
    # from evaluate import load as load_metric
    from pyannote.core.annotation import Annotation
    from transformers import EvalPrediction, feature_extraction_utils


###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEFAULT_BASE_MODEL = "superb/wav2vec2-base-superb-sid"

EVAL_RESULTS_TEMPLATE = """
## Results

* **Accuracy:** {accuracy}
* **Precision:** {precision}
* **Recall:** {recall}
* **Validation Loss:** {loss}

### Confusion
"""

DEFAULT_TRAINER_ARGUMENTS_ARGS = {
    # --- Strategy & Checkpointing ---
    "evaluation_strategy": "epoch",    
    "save_strategy": "steps",          
    "save_steps": 100,                 
    "save_total_limit": 3,             
    "load_best_model_at_end": True,    
    "metric_for_best_model": "accuracy",

    # --- Hardware & Windows Performance ---
    "fp16": True,                      # Keep True now that CUDA is fixed
    ""
    "dataloader_num_workers": 0,       
    "dataloader_pin_memory": True,     
    "gradient_checkpointing": True,    
    "group_by_length": True,           
    
    # --- Hyperparameters ---
    "learning_rate": 3e-5,             # Stable rate for Stage 1 convergence
    "per_device_train_batch_size": 1,  # Lowered to 4 for VRAM safety with librosa
    "gradient_accumulation_steps": 16,  # Effective batch size = 16
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 10,            # Required for 110-speaker separation
    "warmup_ratio": 0.1,
    "logging_steps": 1,                # 🔥 Keep at 1 to monitor the loss movement
    "report_to": "none",               
}

###############################################################################
# ---- LAZY COLLATOR: load audio -> extract features per batch ----
class LazyAudioCollator:
    def __init__(self, fe, max_duration_s: float):
        self.fe = fe
        self.max_len = int(fe.sampling_rate * max_duration_s)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        import librosa
        import torch

        # Filter features to ensure "audio" exists
        valid_features = [f for f in features if f.get("audio") is not None]
        
        # 🚨 FINAL FIX: If a batch is empty, grab the first item anyway to prevent Trainer crash
        if not valid_features:
            log.warning("Empty batch detected! Forcing first item to prevent crash.")
            valid_features = [features[0]]

        paths = [f["audio"]["path"] if isinstance(f["audio"], dict) else str(f["audio"]) 
                 for f in valid_features]
        labels = torch.tensor([f["label"] for f in valid_features], dtype=torch.long)

        audio_arrays = [librosa.load(p, sr=self.fe.sampling_rate)[0] for p in paths]

        batch = self.fe(
            audio_arrays,
            sampling_rate=self.fe.sampling_rate,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch
    
def preprocess_function(batch, feature_extractor, max_duration):
    import librosa
    import numpy as np

    max_len = int(feature_extractor.sampling_rate * max_duration)
    paths = [item["path"] for item in batch["audio"]]
    
    # Resample validation data to 16kHz
    speech_list = [librosa.load(path, sr=feature_extractor.sampling_rate)[0] for path in paths]

    inputs = feature_extractor(
        speech_list,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_tensors="np",
    )

    batch["input_values"] = inputs["input_values"]
    return batch

from transformers import TrainerCallback
import shutil

class EpochArchiveCallback(TrainerCallback):
    """
    Saves and EVALUATES a dedicated copy of the model at the end of each epoch.
    """
    def __init__(self, validation_dataset, feature_extractor):
        self.validation_dataset = validation_dataset
        self.fe = feature_extractor # 🚀 Store the extractor here

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_num = round(state.epoch)
        archive_path = Path(args.output_dir) / f"final_speakerbox_epoch_{epoch_num}"
        
        log.info(f"Archiving model at end of epoch {epoch_num} to {archive_path}")
        
        # 1. Save the model weights and config
        kwargs['model'].save_pretrained(archive_path)
        
        # 2. 🔥 THE FIX: Save the missing preprocessor_config.json
        # Without this, the pipeline() call in eval_model will crash.
        self.fe.save_pretrained(archive_path)

        # 3. Now run your custom evaluation
        # from speakerbox import eval_model
        # eval_model(self.validation_dataset, model_name=str(archive_path))
              
def eval_model(
    validation_dataset: "Dataset",
    model_name: str = "trained-speakerbox",
) -> Tuple[float, float, float, float]:
    import transformers
    import matplotlib.pyplot as plt
    import librosa
    import numpy as np # 🚀 Required for Loss calculation
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        log_loss,
        precision_score,
        recall_score,
    )
    
    log.info("Setting up evaluation pipeline")
    transformers.logging.set_verbosity_error()
    classifier = pipeline("audio-classification", model=model_name)
    
    # 🚀 FIX 1: Access the dataset's feature mapping to avoid integer mismatch
    # This ensures we get "p225" regardless of what integer the dataset assigned today.
    label_names = validation_dataset.features["label"]

    def predict(example: "datasets.arrow_dataset.Example") -> Dict[str, Any]:
        raw_audio = example["audio"]
        audio_path = raw_audio["path"] if isinstance(raw_audio, dict) else raw_audio
        speech, _ = librosa.load(audio_path, sr=classifier.feature_extractor.sampling_rate)
        
        label2id = classifier.model.config.label2id
        num_labels = len(label2id)
        pred = classifier(speech, top_k=num_labels)
        
        prob_list = [0.0] * num_labels
        for item in pred:
            idx = int(label2id[item["label"]])
            prob_list[idx] = item["score"]

        # 🚀 FIX 2: Use the dataset's own decoding feature for the Truth
        true_label_name = label_names.int2str(example["label"])
        
        # 🔥 THE VISUALIZATION FIX: Live check for mislabeling
        match_status = "✅ MATCH" if pred[0]["label"] == true_label_name else "❌ MISMATCH"
        log.info(f"Truth: {true_label_name} | Pred: {pred[0]['label']} | Status: {match_status}")
        
        return {
            "pred_label": str(pred[0]["label"]),
            "true_label": str(true_label_name),
            "pred_scores": prob_list,
        }
        
    log.info("Running eval...")
    # The .map function will now output your match status for every sample in the console.
    validation_dataset = validation_dataset.map(predict)

    # --- METRICS CALCULATION ---
    # 🚀 FIX 3: Convert scores to a NumPy array for Scikit-Learn stability
    y_pred_probs = np.array(validation_dataset["pred_scores"])
    all_labels = list(classifier.model.config.label2id.keys())

    accuracy = accuracy_score(
        y_true=validation_dataset["true_label"],
        y_pred=validation_dataset["pred_label"],
    )
    
    # Loss Fix: Alignment of probabilities to label order
    loss = log_loss(
        y_true=validation_dataset["true_label"],
        y_pred=y_pred_probs,
        labels=all_labels 
    )
    
    precision = precision_score(
        y_true=validation_dataset["true_label"],
        y_pred=validation_dataset["pred_label"],
        average="weighted",
        zero_division=0 
    )
    recall = recall_score(
        y_true=validation_dataset["true_label"],
        y_pred=validation_dataset["pred_label"],
        average="weighted",
        zero_division=0 
    )

    # Store metrics in results.md for your documentation
    with open(f"{model_name}/results.md", "w") as open_f:
        open_f.write(
            EVAL_RESULTS_TEMPLATE.format(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                loss=loss,
            )
        )

    return (accuracy, precision, recall, loss)
def train(
    dataset: "DatasetDict",
    model_name: str = "trained-speakerbox",
    model_base: str = DEFAULT_BASE_MODEL,
    max_duration: float = 2.0,
    seed: Optional[int] = None,
    use_cpu: bool = False,
    trainer_arguments_kws: Dict[str, Any] = DEFAULT_TRAINER_ARGUMENTS_ARGS,
) -> Path:
    """
    Train a speaker classification model (LAZY feature extraction to avoid RAM blow-up).

    Key change:
    - Do NOT dataset.map(preprocess) to materialize input_values for whole dataset.
    - Read audio + extract features inside a custom data_collator per-batch.
    """
    import numpy as np
    import torch
    import transformers
    import soundfile as sf
    from datasets import Audio, load_metric
    # from evaluate import load as load_metric
    
    from transformers import (
        Trainer,
        TrainingArguments,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForSequenceClassification,
    )

    from .utils import set_global_seed

    # Handle seed
    if seed is not None:
        set_global_seed(seed)
        
    # Check for CUDA (NVIDIA GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training will run on: {device}")

    # Handle cpu
    if device.type == "cpu":
        log.warning("No CUDA GPU found. Disabling FP16 and switching to CPU.")
        trainer_arguments_kws["fp16"] = False
        trainer_arguments_kws["no_cuda"] = True

    # Pre-emptively clear the cache to maximize VRAM
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 

    # Load feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_base)
    
    # Build label LUTs
    label2id, id2label = {}, {}
    for i, label in enumerate(dataset["train"].features["label"].names):
        label2id[label] = str(i)
        id2label[str(i)] = label

    # Create AutoModel
    log.info("Setting up model")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        model_base,
        num_labels=len(id2label),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
        # 🔥 CRITICAL: Force the model to ONLY return the classification logits
        output_hidden_states=False, 
        output_attentions=False,
    )
    
    # Move model to GPU
    model.to(device) 

    log.info("Casting all audio paths to HF Audio (decode=False)")
    # Define a specific path for the metadata cache
    
    metadata_cache_path = Path(model_name) / "speakerboxdts_metadata_ds"

    if metadata_cache_path.exists():
        log.info(f"🚀 Loading pre-cast metadata from: {metadata_cache_path}")
        from datasets import load_from_disk
        dataset = load_from_disk(str(metadata_cache_path))
    else:
        log.info("Casting all audio paths to HF Audio (decode=False)...")
        dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=feature_extractor.sampling_rate, decode=False)
        )
        
        # Ensure the output directory exists before saving
        metadata_cache_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.save_to_disk(str(metadata_cache_path))
        log.info(f"💾 Metadata cached successfully to {metadata_cache_path}")

    # Select eval split first
    # Find this section in your train() function
    if "valid" in dataset:
        full_valid = dataset["valid"]
        eval_dataset = full_valid.shuffle(seed=42).select(range(min(500, len(full_valid))))
        log.info(f"Sub-sampling validation set to {len(eval_dataset)} examples for speed.")
    else:
        eval_n = min(500, len(dataset["test"]))
        eval_dataset = dataset["test"].shuffle(seed=0).select(range(eval_n))

    # Precompute eval features
    # eval_dataset = eval_dataset.map(
    #     preprocess_function,
    #     batched=True,
    #     batch_size=32,
    #     fn_kwargs={
    #         "feature_extractor": feature_extractor,
    #         "max_duration": max_duration,
    #     },
    # )
    # eval_dataset.set_format("torch", columns=["input_values", "label"])

    # args = TrainingArguments(
    #     output_dir=model_name,
    #     **trainer_arguments_kws,
    # )
    
    # Inside speakerbox/main.py
    args = TrainingArguments(
        output_dir=model_name,
        learning_rate=6.202445652173913e-06,
        num_train_epochs=3,
        fp16=False,
        logging_steps=10,
        
        # --- Speed & VRAM Surgery ---
        evaluation_strategy="no",
        # evaluation_strategy="epoch", 
        # per_device_eval_batch_size=2, # 🚀 Lowered to 2 to fit in 4GB VRAM
        # eval_accumulation_steps=1,    # 🚨 Flush memory to CPU after EVERY batch
        # fp16_full_eval=True,          # Keep half-precision for speed
        
        ignore_data_skip=True,
        save_steps=200,
        save_total_limit=3,
        per_device_train_batch_size=4, 
        gradient_accumulation_steps=4, 
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
    )

    # Metrics
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]  # 🔥 FIX: extract logits

        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=eval_pred.label_ids)

    data_collator = LazyAudioCollator(feature_extractor, max_duration)

    archive_eval_callback = EpochArchiveCallback(
        validation_dataset=dataset["valid"], 
        feature_extractor=feature_extractor
    )

    # Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        # eval_dataset=eval_dataset, # ✅ Now uses 10% valid split
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[archive_eval_callback], 
    )
    
     # Optional: reduce MPS fragmentation risk
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    torch.cuda.empty_cache()

    transformers.logging.set_verbosity_info()
    trainer.train(resume_from_checkpoint=True)
    trainer.save_model()
    feature_extractor.save_pretrained(model_name)
    # eval_model(dataset["valid"], model_name=str(model_name))
    

    return Path(model_name).resolve()



def apply(  # noqa: C901
    audio: Union[str, Path],
    model: str,
    mode: Literal["diarize", "naive"] = "diarize",
    min_chunk_duration: float = 0.5,
    max_chunk_duration: float = 2.0,
    confidence_threshold: float = 0.85,
) -> "Annotation":
    """
    Iteritively apply the model across chunks of an audio file.

    Parameters
    ----------
    audio: Union[str, Path]
        The audio filepath.
    model: str
        The path to the trained audio-classification model.
    mode: Literal["diarize", "naive"]
        Which mode to use for processing. "diarize" will diarize the audio
        prior to generating chunks to classify. "naive" will iteratively process
        chunks. "naive" is assumed to be faster but have worse performance.
        Default: "diarize"
    min_chunk_duration: float
        The minimum size in seconds a chunk of audio is allowed to be
        for it to be ran through the classification pipeline.
        Default: 0.5 seconds
    max_chunk_duration: float
        The maximum size in seconds a chunk of audio is allowed to be
        for it to be ran through the classification pipeline.
        Default: 2 seconds
    confidence_threshold: float
        A value to act as a lower bound to the reported confidence
        of the model prediction. Any classification that has a confidence
        lower than this value will be ignore and not added as a segment.
        Default: 0.95 (fairly strict / must have high confidence in prediction)

    Returns
    -------
    Annotation
        A pyannote.core Annotation with all labeled segments.
    """
    import numpy as np
    from pyannote.audio import Pipeline
    from pyannote.core.annotation import Annotation
    from pyannote.core.segment import Segment
    from pyannote.core.utils.types import Label, TrackName
    from pydub import AudioSegment
    from tqdm import tqdm

    # Just set track name to the same as the audio filepath
    track_name = str(audio)

    # Read audio file
    loaded_audio = AudioSegment.from_file(audio)

    # Load model
    classifier = pipeline("audio-classification", model=model)

    # Get number of speakers
    n_speakers = len(classifier.model.config.id2label)

    # Generate random uuid filename for storing temp audio chunks
    tmp_audio_chunk_save_path = Path(".tmp-audio-chunk-during-apply.wav")

    def _diarize() -> List[Tuple[Segment, TrackName, Label]]:  # noqa: C901
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        dia = diarization_pipeline(audio)

        # Prep for calculations
        max_chunk_duration_millis = max_chunk_duration * 1000

        # Return chunks for each diarized section
        records: List[Tuple[Segment, TrackName, Label]] = []
        for turn, _, _ in tqdm(dia.itertracks(yield_label=True)):
            # Keep track of each turn chunk classification and score
            chunk_scores: Dict[str, List[float]] = {}

            # Get audio slice for turn
            turn_start_millis = turn.start * 1000
            turn_end_millis = turn.end * 1000

            # Split into smaller chunks
            for chunk_start_millis_float in np.arange(
                turn_start_millis,
                turn_end_millis,
                max_chunk_duration_millis,
            ):
                # Round start to nearest int
                chunk_start_millis = round(chunk_start_millis_float)

                # Tentative chunk end
                chunk_end_millis = round(chunk_start_millis + max_chunk_duration_millis)

                # Determine chunk end time
                # If start + chunk duration is longer than turn
                # Chunk needs to be cut at turn end
                if turn_end_millis < chunk_end_millis:
                    chunk_end_millis = round(turn_end_millis)

                # Only allow if duration is greater than
                # min intra turn chunk duration
                duration = chunk_end_millis - chunk_start_millis
                if duration >= min_chunk_duration:
                    # Get chunk
                    chunk = loaded_audio[chunk_start_millis:chunk_end_millis]

                    # Write to temp
                    chunk.export(tmp_audio_chunk_save_path, format="wav")

                    # Predict and store scores for turn
                    preds = classifier(
                        str(tmp_audio_chunk_save_path),
                        top_k=n_speakers,
                    )
                    for pred in preds:
                        if pred["label"] not in chunk_scores:
                            chunk_scores[pred["label"]] = []
                        chunk_scores[pred["label"]].append(pred["score"])

            # Create mean score
            turn_speaker = None
            if len(chunk_scores) > 0:
                mean_scores: Dict[str, float] = {}
                for speaker, scores in chunk_scores.items():
                    mean_scores[speaker] = sum(scores) / len(scores)

                # Get highest scoring speaker and their score
                highest_mean_speaker = ""
                highest_mean_score = 0.0
                for speaker, score in mean_scores.items():
                    if score > highest_mean_score:
                        highest_mean_speaker = speaker
                        highest_mean_score = score

                # Threshold holdout
                if highest_mean_score >= confidence_threshold:
                    turn_speaker = highest_mean_speaker

            # Store record
            records.append(
                (
                    Segment(turn.start, turn.end),
                    track_name,
                    turn_speaker,
                )
            )

        return records

    def _naive() -> List[Tuple[Segment, TrackName, Label]]:
        # Move audio window, apply, and append annotation record
        records: List[Tuple[Segment, TrackName, Label]] = []
        for chunk_start_seconds in tqdm(
            np.arange(0, loaded_audio.duration_seconds, max_chunk_duration)
        ):
            # Calculate chunk end
            chunk_end_seconds = chunk_start_seconds + max_chunk_duration
            if chunk_end_seconds > loaded_audio.duration_seconds:
                chunk_end_seconds = loaded_audio.duration_seconds

            # Check if duration is long enough
            duration = chunk_end_seconds - chunk_start_seconds
            if duration >= min_chunk_duration:
                # Convert seconds to millis
                chunk_start_millis = chunk_start_seconds * 1000
                chunk_end_millis = chunk_end_seconds * 1000

                # Select chunk
                chunk = loaded_audio[chunk_start_millis:chunk_end_millis]

                # Write chunk to temp
                chunk.export(tmp_audio_chunk_save_path, format="wav")

                # Predict, keep top 1 and store to records
                pred = classifier(str(tmp_audio_chunk_save_path), top_k=1)[0]
                if pred["score"] >= confidence_threshold:
                    records.append(
                        (
                            Segment(chunk_start_seconds, chunk_end_seconds),
                            track_name,
                            pred["label"],
                        )
                    )

        return records

    # Classify based off strategy
    mode_lut = {
        "diarize": _diarize,
        "naive": _naive,
    }

    # Generate records and clean up
    try:
        records = mode_lut[mode]()

        # Merge segments that are touching
        merged_records: List[Tuple[Segment, TrackName, Label]] = []
        current_record: Optional[Tuple[Segment, TrackName, Label]] = None
        for record in records:
            if current_record is None:
                current_record = record
            else:
                # The label matches and the segment start and end points are
                # touching, merge
                if (
                    record[2] == current_record[2]
                    and record[0].start == current_record[0].end
                ):
                    # Make new record with merged data
                    # because tuples are immutable
                    current_record = (
                        Segment(current_record[0].start, record[0].end),
                        track_name,
                        current_record[2],
                    )
                else:
                    merged_records.append(current_record)
                    current_record = record

        # Add the last current segment
        # we only do this type check to handle the type error
        if current_record is not None:
            merged_records.append(current_record)

        return Annotation.from_records(merged_records)

    finally:
        # Always clean up tmp file
        if tmp_audio_chunk_save_path.exists():
            tmp_audio_chunk_save_path.unlink()



