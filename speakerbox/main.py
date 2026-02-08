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
    from evaluate import load as load_metric
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
    "eval_strategy": "epoch",
    "save_strategy": "steps",
    "save_steps": 200,
    "learning_rate": 3e-5,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 1,
    "eval_accumulation_steps": 1,
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 5,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "gradient_checkpointing": False,
}

###############################################################################
# ---- LAZY COLLATOR: load audio -> extract features per batch ----
class LazyAudioCollator:
    def __init__(self, fe, max_duration_s: float):
        self.fe = fe
        self.max_len = int(fe.sampling_rate * max_duration_s)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        import soundfile as sf
        import numpy as np
        import torch

        paths = []
        valid_labels = []
        
        for f in features:
            audio_data = f.get("audio")
            path = None
            
            # Extract path from dictionary or string safely
            if isinstance(audio_data, dict):
                path = audio_data.get("path")
            elif audio_data is not None:
                path = str(audio_data)
            
            # Only add to the batch if we actually found a file path
            if path:
                paths.append(path)
                valid_labels.append(f["label"])
            else:
                log.warning(f"Skipping empty audio path for speaker {f.get('label')}")

        labels = torch.tensor(valid_labels, dtype=torch.long)

        audio_arrays: List[np.ndarray] = []
        for p in paths:
            # sf.read is faster than librosa for raw loading on macOS
            wav, _ = sf.read(p)
            audio_arrays.append(wav)

        batch = self.fe(
            audio_arrays,
            sampling_rate=self.fe.sampling_rate,
            max_length=self.max_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        batch["labels"] = labels
        return batch
    
# ---- PRECOMPUTE FEATURES FOR FAST EVAL ----
def preprocess_function(batch, feature_extractor, max_duration):
    import soundfile as sf
    import numpy as np

    max_len = int(feature_extractor.sampling_rate * max_duration)

    paths = [item["path"] for item in batch["audio"]]
    
    speech_list = []
    for path in paths:
        wav, sr = sf.read(path)
        speech_list.append(wav)

    inputs = feature_extractor(
        speech_list,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=max_len,
        truncation=True,
        padding=True,
        return_tensors="np",
    )

    batch["input_values"] = inputs["input_values"]
    return batch

from transformers import TrainerCallback
import shutil

class EpochArchiveCallback(TrainerCallback):
    """
    Saves a dedicated copy of the model in a new folder at the end of each epoch.
    """
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_num = round(state.epoch)
        # Define a unique folder name for this epoch
        archive_path = Path(args.output_dir) / f"archive_epoch_{epoch_num}"
        
        log.info(f"Archiving model at end of epoch {epoch_num} to {archive_path}")
        
        # Save the model and feature extractor specifically to this folder
        kwargs['model'].save_pretrained(archive_path)
        if 'processing_class' in kwargs: # for newer transformers
            kwargs['processing_class'].save_pretrained(archive_path)
        elif 'feature_extractor' in kwargs:
            kwargs['feature_extractor'].save_pretrained(archive_path)

def eval_model(
    validation_dataset: "Dataset",
    model_name: str = "trained-speakerbox",
) -> Tuple[float, float, float, float]:
    """
    Evaluate a trained model.

    This will store two files in the model directory, one for the accuracy, precision,
    and recall in a markdown file and the other is the generated top one confusion
    matrix as a PNG file.

    Parameters
    ----------
    validation_dataset: Dataset
        The dataset to validate the model against.
    model_name: str
        A name for the model. This will also create a directory with the same name
        to store the produced model in.
        Default: "trained-speakerbox"

    Returns
    -------
    accuracy: float
        The model accuracy as returned by sklearn.metrics.accuracy_score.
    precision: float
        The model (weighted) precision as returned by sklearn.metrics.precision_score.
    recall: float
        The model (weighted) recall as returned by sklearn.metrics.recall_score.
    loss: float
        The model log loss as returned by sklearn.metrics.log_loss.
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        log_loss,
        precision_score,
        recall_score,
    )

    log.info("Setting up evaluation pipeline")
    classifier = pipeline(
        "audio-classification",
        model=model_name,
    )
    log.info("Running eval")


    import librosa
    import torch
    import torch.nn.functional as F    
    from typing import Dict, Any

    def predict(example: "datasets.arrow_dataset.Example") -> Dict[str, Any]:
        audio_path = example["audio"]
        speech, _ = librosa.load(audio_path, sr=classifier.feature_extractor.sampling_rate)
        
        # Get total labels from config
        label2id = classifier.model.config.label2id
        num_labels = len(label2id)
        
        # Get predictions from pipeline
        pred = classifier(speech, top_k=num_labels)
        
        # CRITICAL: Create an empty list of zeros for the full distribution
        # This ensures every speaker has a slot, even if the model gave it 0.0
        prob_list = [0.0] * num_labels
        
        for item in pred:
            label_str = item["label"]
            score = item["score"]
            # Use the config's mapping to place the score in the correct index
            idx = int(label2id[label_str])
            prob_list[idx] = score

        # Normalize to ensure sum is exactly 1.0 (prevents sklearn warnings)
        total = sum(prob_list)
        prob_list = [p / total for p in prob_list]
        
        return {
            "pred_label": pred[0]["label"],
            "true_label": classifier.model.config.id2label[example["label"]],
            "pred_scores": prob_list,
        }
        
    validation_dataset = validation_dataset.map(predict)

    # Create confusion
    ConfusionMatrixDisplay.from_predictions(
        validation_dataset["true_label"],
        validation_dataset["pred_label"],
    )
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.savefig(f"{model_name}/validation-confusion.png", bbox_inches="tight")

    # Compute metrics
    accuracy = accuracy_score(
        y_true=validation_dataset["true_label"],
        y_pred=validation_dataset["pred_label"],
    )
    precision = precision_score(
        y_true=validation_dataset["true_label"],
        y_pred=validation_dataset["pred_label"],
        average="weighted",
    )
    recall = recall_score(
        y_true=validation_dataset["true_label"],
        y_pred=validation_dataset["pred_label"],
        average="weighted",
    )
    loss = log_loss(
        y_true=validation_dataset["true_label"],
        y_pred=validation_dataset["pred_scores"],
    )

    # Store metrics
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
    from datasets import Audio
    from evaluate import load as load_metric
    
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

    # Handle cpu
    if use_cpu:
        trainer_arguments_kws["no_cuda"] = True

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
    )


    log.info("Casting all audio paths to HF Audio (decode=False)")
    dataset = dataset.cast_column(
        "audio", Audio(sampling_rate=feature_extractor.sampling_rate, decode=False)
    )

    # Select eval split first
    if "valid" in dataset:
        eval_dataset = dataset["valid"].shuffle(seed=42).select(range(min(500, len(dataset["valid"]))))
        log.info(f"Sub-sampling validation set to {len(eval_dataset)} examples for speed.")
    else:
        eval_n = min(500, len(dataset["test"]))
        eval_dataset = dataset["test"].shuffle(seed=0).select(range(eval_n))

    # Precompute eval features
    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=32,
        fn_kwargs={
            "feature_extractor": feature_extractor,
            "max_duration": max_duration,
        },
    )
    eval_dataset.set_format("torch", columns=["input_values", "label"])

    # args = TrainingArguments(
    #     output_dir=model_name,
    #     **trainer_arguments_kws,
    # )
    
    args = TrainingArguments(
        output_dir=model_name,
        learning_rate=3e-5,
        num_train_epochs=10,             
        
        use_mps_device=True,            
        fp16=False,                      
        bf16=False,                      
        dataloader_num_workers=0,        
        dataloader_pin_memory=False,     

        eval_strategy="epoch",
        per_device_eval_batch_size=4,    
        eval_accumulation_steps=1,       

        per_device_train_batch_size=4,   
        gradient_accumulation_steps=4,   # Effective batch size = 16
        logging_steps=10,
        save_strategy="steps",           # Save once per epoch for your archive folders
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
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

    # Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=eval_dataset, # ✅ Now uses 10% valid split
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EpochArchiveCallback()],
    )
    
     # Optional: reduce MPS fragmentation risk
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    torch.cuda.empty_cache()

    transformers.logging.set_verbosity_info()
    trainer.train(resume_from_checkpoint=False)
    trainer.save_model()
    feature_extractor.save_pretrained(model_name)

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



