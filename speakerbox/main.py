#!/usr/bin/env python

import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from transformers import TrainerCallback, pipeline

if TYPE_CHECKING:
    from datasets import Audio, Dataset, DatasetDict
    from pyannote.core.annotation import Annotation
    from transformers import EvalPrediction

###############################################################################

log = logging.getLogger(__name__)

EVAL_RESULTS_TEMPLATE = """
## Results

* **Accuracy:** {accuracy}
* **Precision:** {precision}
* **Recall:** {recall}
* **Validation Loss:** {loss}

### Confusion
"""

###############################################################################
# ── SECTION 1: AUDIO LOADING ──────────────────────────────────────────────────

class LazyAudioCollator:
    """
    Loads raw audio from disk and extracts features lazily per batch.
    Avoids pre-materialising the full dataset into RAM.

    Parameters
    ----------
    fe             : Wav2Vec2FeatureExtractor
    max_duration_s : float — clips longer than this are truncated
    audio_backend  : str   — "soundfile" (fast) or "librosa" (auto-resamples)
    """

    def __init__(self, fe, max_duration_s: float, audio_backend: str = "soundfile"):
        self.fe           = fe
        self.max_len      = int(fe.sampling_rate * max_duration_s)
        self.audio_backend = audio_backend

    def _load(self, path: str) -> np.ndarray:
        if self.audio_backend == "librosa":
            import librosa
            return librosa.load(path, sr=self.fe.sampling_rate)[0]
        else:
            import soundfile as sf
            # print(f"\nLoading audio from: {path}\n")
            wav, _ = sf.read(path)
            return wav

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        valid = [f for f in features if f.get("audio") is not None]
        if not valid:
            log.warning("Empty batch — forcing first item to prevent crash.")
            valid = [features[0]]

        paths  = [
            f["audio"]["path"] if isinstance(f["audio"], dict) else str(f["audio"])
            for f in valid
        ]
        labels = torch.tensor([f["label"] for f in valid], dtype=torch.long)
        audio_arrays = [self._load(p) for p in paths]

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


def preprocess_function(
    batch,
    feature_extractor,
    max_duration: float,
    audio_backend: str = "soundfile",
):
    """
    Precompute input_values for a batch (used for eval pre-caching).

    Parameters
    ----------
    audio_backend : str
        "soundfile" (default) or "librosa" (auto-resamples to target sr).
    """
    max_len = int(feature_extractor.sampling_rate * max_duration)
    paths = [item["path"] for item in batch["audio"]]

    if audio_backend == "librosa":
        import librosa
        speech_list = [librosa.load(p, sr=feature_extractor.sampling_rate)[0] for p in paths]
    else:
        import soundfile as sf
        speech_list = [sf.read(p)[0] for p in paths]

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


###############################################################################

class EpochArchiveCallback(TrainerCallback):
    """
    At the end of every epoch:
      1. Copies latest HF checkpoint → <output_dir>/final_speakerbox_epoch_N
      2. Saves feature extractor into that folder
      3. Optionally runs eval_model on the validation set
    """

    def __init__(
        self,
        validation_dataset,
        feature_extractor,
        run_eval:      bool = True,
        eval_mode:     str  = "softmax",
        audio_backend: str  = "soundfile",
    ):
        self.validation_dataset = validation_dataset
        self.fe            = feature_extractor
        self.run_eval      = run_eval
        self.eval_mode     = eval_mode
        self.audio_backend = audio_backend

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_num    = int(round(state.epoch))
        output_dir   = Path(args.output_dir)
        archive_path = output_dir / f"final_speakerbox_epoch_{epoch_num}"
        log.info(f"Archiving checkpoint — epoch {epoch_num} → {archive_path}")

        checkpoints = sorted(output_dir.glob("checkpoint-*"), key=lambda p: p.stat().st_mtime)
        if not checkpoints:
            log.warning("No checkpoint-* folder found — skipping archive.")
            return control

        if archive_path.exists():
            shutil.rmtree(archive_path)
        shutil.copytree(checkpoints[-1], archive_path)

        kwargs["model"].save_pretrained(archive_path)
        self.fe.save_pretrained(archive_path)

        if self.run_eval:
            eval_model(
                self.validation_dataset,
                model_name=str(archive_path),
                eval_mode=self.eval_mode,
                audio_backend=self.audio_backend,
            )
        return control


###############################################################################
# ── SECTION 3: EVALUATION ─────────────────────────────────────────────────────

def eval_model(
    validation_dataset: "Dataset",
    model_name:    str = "trained-speakerbox",
    eval_mode:     str = "softmax",
    audio_backend: str = "soundfile",
) -> Tuple[float, float, float, float]:
    """
    Evaluate a trained speaker-classification model.

    Parameters
    ----------
    validation_dataset : Dataset — HF Dataset with "audio" and "label" columns
    model_name         : str     — path to saved model directory
    eval_mode          : str     — "softmax" (batched, mac-friendly) or
                                   "pipeline" (HF pipeline, windows-style)
    audio_backend      : str     — "soundfile" or "librosa" (softmax mode only)
    """
    import transformers
    from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score

    transformers.logging.set_verbosity_error()

    # ── pipeline mode ──────────────────────────────────────────────────────── #
    if eval_mode == "pipeline":
        import librosa
        from transformers import pipeline as hf_pipeline

        log.info("Setting up evaluation (pipeline mode)")
        classifier  = hf_pipeline("audio-classification", model=model_name)
        label_names = validation_dataset.features["label"]
        label2id    = classifier.model.config.label2id
        num_labels  = len(label2id)

        def _predict_one(example):
            raw    = example["audio"]
            path   = raw["path"] if isinstance(raw, dict) else raw
            speech, _ = librosa.load(path, sr=classifier.feature_extractor.sampling_rate)

            preds     = classifier(speech, top_k=num_labels)
            prob_list = [0.0] * num_labels
            for item in preds:
                prob_list[int(label2id[item["label"]])] = item["score"]

            true_name = label_names.int2str(example["label"])
            # match     = "✅" if preds[0]["label"] == true_name else "❌"
            # log.info(f"Truth: {true_name} | Pred: {preds[0]['label']} | {match}")
            return {
                "pred_label":  str(preds[0]["label"]),
                "true_label":  str(true_name),
                "pred_scores": prob_list,
            }

        log.info("Running eval (pipeline mode)...")
        validation_dataset = validation_dataset.map(_predict_one)
        y_pred_probs = np.array(validation_dataset["pred_scores"])
        all_labels   = list(label2id.keys())

    # ── softmax mode ───────────────────────────────────────────────────────── #
    else:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        log.info(f"Setting up evaluation (softmax mode, device={device})")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name).to(device)
        model.eval()
        fe = Wav2Vec2FeatureExtractor.from_pretrained(model_name)

        id2label       = model.config.id2label
        label2id       = model.config.label2id
        all_labels     = [id2label[i] for i in range(len(id2label))]
        dataset_labels = list(validation_dataset.features["label"].names)
        max_len        = int(fe.sampling_rate * 2.0)

        @torch.no_grad()
        def _predict_batch(batch):
            paths = [a["path"] if isinstance(a, dict) else str(a) for a in batch["audio"]]
            if audio_backend == "librosa":
                import librosa
                wavs = [librosa.load(p, sr=fe.sampling_rate)[0] for p in paths]
            else:
                import soundfile as sf
                wavs = [sf.read(p)[0] for p in paths]

            inputs  = fe(wavs, sampling_rate=fe.sampling_rate, max_length=max_len,
                         truncation=True, padding=True, return_tensors="pt")
            inputs  = {k: v.to(device) for k, v in inputs.items()}
            probs   = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
            pred_ids = np.argmax(probs, axis=-1)
            return {
                "pred_label":  [id2label[int(i)] for i in pred_ids],
                "true_label":  [dataset_labels[x] for x in batch["label"]],
                "pred_scores": probs.tolist(),
            }

        log.info("Running eval (softmax mode)...")
        validation_dataset = validation_dataset.map(
            _predict_batch, batched=True, batch_size=16, load_from_cache_file=False
        )
        y_pred_probs = np.array(validation_dataset["pred_scores"])

    # ── shared metrics ─────────────────────────────────────────────────────── #
    y_true = validation_dataset["true_label"]
    y_pred = validation_dataset["pred_label"]

    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    loss      = log_loss(y_true, y_pred_probs, labels=all_labels)

    os.makedirs(model_name, exist_ok=True)
    with open(f"{model_name}/results.md", "w") as f:
        f.write(EVAL_RESULTS_TEMPLATE.format(
            accuracy=accuracy*100, 
            precision=precision*100, 
            recall=recall*100, 
            loss=loss,
        ))
    log.info(f"Eval results saved → {model_name}/results.md")

    return accuracy, precision, recall, loss


###############################################################################
# ── SECTION 4: CHECKPOINT HELPER ──────────────────────────────────────────────

def _find_last_checkpoint(output_dir: str) -> Optional[str]:
    """
    Returns path of the latest checkpoint-* folder, or None if none exist.
    Passing None to trainer.train() starts fresh without crashing.
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    checkpoints = sorted(
        output_path.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if not checkpoints:
        return None
    log.info(f"Resuming from checkpoint: {checkpoints[-1]}")
    return str(checkpoints[-1])


def _build_scratch_model(
    num_labels:  int,
    label2id:    Dict[str, str],
    id2label:    Dict[str, str],
    model_base:  str = "superb/wav2vec2-base-superb-sid",
) -> "torch.nn.Module":
    """
    Build a Wav2Vec2ForSequenceClassification with RANDOM weights.

    Uses the architecture config from model_base (hidden size, layers, etc.)
    but discards all pretrained weights — every parameter is randomly
    initialised from scratch.

    Parameters
    ----------
    num_labels : int          — number of speaker classes
    label2id   : dict         — speaker name → int index
    id2label   : dict         — int index → speaker name
    model_base : str          — HF model ID used only to borrow the config
                                (no weights are downloaded beyond the config)
    """
    from transformers import Wav2Vec2Config, Wav2Vec2ForSequenceClassification

    config = Wav2Vec2Config.from_pretrained(model_base)

    config.num_labels  = num_labels
    config.label2id    = label2id
    config.id2label    = id2label
    
    model = Wav2Vec2ForSequenceClassification(config)

    log.info(
        f"Built from-scratch Wav2Vec2ForSequenceClassification "
        f"({config.num_hidden_layers} layers, hidden={config.hidden_size}, "
        f"num_labels={num_labels}) — all weights randomly initialised."
    )
    return model


###############################################################################
# ── SECTION 6: TRAINING ───────────────────────────────────────────────────────

def train(
    dataset:               "DatasetDict",
    model_name:            str,
    model_base:            str,
    max_duration:          float,
    trainer_arguments_kws: Dict[str, Any],
    eval_mode:             str            = "softmax",
    audio_backend:         str            = "soundfile",
    metadata_cache_path:   Optional[str]  = None,
    seed:                  Optional[int]  = None,
    resume_from_checkpoint: bool          = True,
    train_mode:            str            = "finetune",
) -> Path:
    """
    Train a speaker classification model with lazy per-batch feature extraction.

    Parameters
    ----------
    dataset               : DatasetDict — must contain "train" and "valid" splits
    model_name            : str         — output directory for the trained model
    model_base            : str         — HF model ID used for fine-tuning weights
                                          OR (train_mode="scratch") config only
    max_duration          : float       — max clip length in seconds
    trainer_arguments_kws : dict        — passed directly to HF TrainingArguments
    eval_mode             : str         — "softmax" or "pipeline"
    audio_backend         : str         — "soundfile" or "librosa"
    metadata_cache_path   : str or None — cache for Audio-cast dataset;
                                          None=auto, ""=disable
    seed                  : int or None — global random seed
    resume_from_checkpoint: bool        — True=auto-resume, False=always fresh
    train_mode            : str
        "finetune" (default) — load pretrained weights from model_base and
                               fine-tune the whole network on your speakers.
                               Fast convergence, recommended when you have
                               limited data (<500 h).

        "scratch"            — initialise all wav2vec2 weights randomly using
                               only the architecture config from model_base.
                               No pretrained weights are used at all.
                               Requires much more data and compute to converge,
                               but gives full control and no licence constraints.
                               Uses Wav2Vec2ForSequenceClassification built
                               from Wav2Vec2Config (equivalent to the
                               Wav2Vec2ForPreTraining path but with a
                               classification head already attached).
    """
    import transformers
    from datasets import Audio
    from evaluate import load as load_metric  
    from transformers import (
        Trainer,
        TrainingArguments,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2ForSequenceClassification,
    )
    from .utils import set_global_seed

    if seed is not None:
        set_global_seed(seed)

    # ── device ────────────────────────────────────────────────────────────── #
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        torch.mps.empty_cache()
    else:
        device = torch.device("cpu")
        log.warning("No GPU found — disabling fp16 and forcing CPU.")
        trainer_arguments_kws["fp16"]    = False
        trainer_arguments_kws["no_cuda"] = True
    log.info(f"Training on: {device} | train_mode: {train_mode} | eval_mode: {eval_mode}")

    # ── feature extractor ─────────────────────────────────────────────────── #
    # Always loaded from model_base — it is a small config/vocab file and is
    # needed in both finetune and scratch modes so the collator knows the SR.
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_base)

    # ── label maps ────────────────────────────────────────────────────────── #
    label2id, id2label = {}, {}
    for i, label in enumerate(dataset["train"].features["label"].names):
        label2id[label]  = str(i)
        id2label[str(i)] = label
    num_labels = len(id2label)

    # ── model ─────────────────────────────────────────────────────────────── #
    if train_mode == "scratch":
        # ── FROM SCRATCH ─────────────────────────────────────────────────── #
        # Random weights — only the architecture config is borrowed from
        # model_base.  No checkpoint weights are downloaded or used.
        model = _build_scratch_model(
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            model_base=model_base,
        ).to(device)
    else:
        # ── FINE-TUNE ─────────────────────────────────────────────────────── #
        # Load full pretrained weights; replace the classification head to
        # match the number of speakers in this dataset.
        log.info(f"Loading pretrained weights from: {model_base}")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_base,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True,
        ).to(device)

    # ── audio metadata cache ──────────────────────────────────────────────── #
    # print(f"\ndata set: {dataset['train'][0]['audio']['path']}\n")
    if metadata_cache_path == "":
        # print(f"\n1\n")
        log.info("Metadata caching disabled — casting on-the-fly.")
        dataset = dataset.cast_column(
            "audio", Audio(sampling_rate=feature_extractor.sampling_rate, decode=False)
        )
    else:
        # print(f"\nmetdata: {metadata_cache_path}\n")
        cache = (
            Path(metadata_cache_path) if metadata_cache_path
            else Path(model_name) / f"{Path(model_name).name}_metadata_ds"
        )
        # print(f"\ncache: {cache}\n")
        if cache.exists():
            # print(f"\n1\n")
            log.info(f"Loading cached metadata from: {cache}")
            from datasets import load_from_disk
            # print(f"\ndata set: {dataset['train'][0]['audio']['path']}\n")
            # dataset = load_from_disk(str(cache))
            # print(f"\ndata set: {dataset['train'][0]['audio']['path']}\n")
        else:
            # print(f"\n2\n")
            log.info("Casting audio paths (decode=False) and caching...")
            # print(f"\ndata set: {dataset['train'][0]['audio']['path']}\n")
            dataset = dataset.cast_column(
                "audio", Audio(sampling_rate=feature_extractor.sampling_rate, decode=False)
            )
            cache.parent.mkdir(parents=True, exist_ok=True)
            # print(f"\ndata set: {dataset['train'][0]['audio']['path']}\n")
            dataset.save_to_disk(str(cache))
            log.info(f"Metadata cached → {cache}")

    # ── training arguments ────────────────────────────────────────────────── #
    # print("Training Arguments:")
    # for k, v in trainer_arguments_kws.items():
    #     print(f"- {k}: {v}")
    # print("before save_steps:", trainer_arguments_kws.get("save_steps"))
    args   = TrainingArguments(output_dir=model_name, **trainer_arguments_kws)
    # print("after save_steps:", trainer_arguments_kws.get("save_steps"))

    metric = load_metric("accuracy")
    # print(f"\ndata set: {dataset['train'][0]['audio']['path']}\n")

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=eval_pred.label_ids)

    # ── collator + callback ───────────────────────────────────────────────── #
    data_collator    = LazyAudioCollator(feature_extractor, max_duration, audio_backend=audio_backend)
    archive_callback = EpochArchiveCallback(
        validation_dataset=dataset["valid"],
        feature_extractor=feature_extractor,
        audio_backend=audio_backend,
        eval_mode=eval_mode,
    )
    # print(f"\ndata set: {dataset['train'][0]['audio']['path']}\n")
    # ── trainer ───────────────────────────────────────────────────────────── #
    # print("before trainer save_steps:", trainer_arguments_kws.get("save_steps"))
    # print(f"argument: {args}")
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[archive_callback],
    )
    print("before checkpoint:", trainer_arguments_kws.get("save_steps"))
    transformers.logging.set_verbosity_info()
    if resume_from_checkpoint:
        checkpoint = _find_last_checkpoint(model_name) 
        trainer.train(resume_from_checkpoint=checkpoint)
    else :
        trainer.train(resume_from_checkpoint=None)
    trainer.save_model()
    feature_extractor.save_pretrained(model_name)

    return Path(model_name).resolve()


###############################################################################
# ── SECTION 7: APPLY ──────────────────────────────────────────────────────────

def apply(  # noqa: C901
    audio: Union[str, Path],
    model: str,
    mode: Literal["diarize", "naive"] = "diarize",
    min_chunk_duration: float = 0.5,
    max_chunk_duration: float = 2.0,
    confidence_threshold: float = 0.85,
) -> "Annotation":
    """
    Iteratively apply the model across chunks of an audio file.

    Parameters
    ----------
    audio : Union[str, Path]
        Path to the audio file.
    model : str
        Path to the trained audio-classification model.
    mode : Literal["diarize", "naive"]
        "diarize" — diarizes first then classifies each turn (slower, better).
        "naive"   — slides a fixed window and classifies each chunk (faster).
    min_chunk_duration : float
        Minimum chunk size in seconds to be classified. Default: 0.5
    max_chunk_duration : float
        Maximum chunk size in seconds. Default: 2.0
    confidence_threshold : float
        Predictions below this confidence are discarded. Default: 0.85

    Returns
    -------
    Annotation
        pyannote.core Annotation with all labeled segments.
    """
    import numpy as np
    from pyannote.audio import Pipeline
    from pyannote.core.annotation import Annotation
    from pyannote.core.segment import Segment
    from pyannote.core.utils.types import Label, TrackName
    from pydub import AudioSegment
    from tqdm import tqdm

    track_name = str(audio)
    loaded_audio = AudioSegment.from_file(audio)
    classifier = pipeline("audio-classification", model=model)
    n_speakers = len(classifier.model.config.id2label)
    tmp_path = Path(".tmp-audio-chunk-during-apply.wav")

    def _diarize() -> List[Tuple[Segment, TrackName, Label]]:  # noqa: C901
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
        dia = diarization_pipeline(audio)
        max_ms = max_chunk_duration * 1000
        records: List[Tuple[Segment, TrackName, Label]] = []

        for turn, _, _ in tqdm(dia.itertracks(yield_label=True)):
            chunk_scores: Dict[str, List[float]] = {}
            t_start_ms = turn.start * 1000
            t_end_ms = turn.end * 1000

            for cs_float in np.arange(t_start_ms, t_end_ms, max_ms):
                cs = round(cs_float)
                ce = round(cs + max_ms)
                if t_end_ms < ce:
                    ce = round(t_end_ms)
                if (ce - cs) >= min_chunk_duration:
                    loaded_audio[cs:ce].export(tmp_path, format="wav")
                    for pred in classifier(str(tmp_path), top_k=n_speakers):
                        chunk_scores.setdefault(pred["label"], []).append(pred["score"])

            if chunk_scores:
                mean_scores = {spk: sum(s) / len(s) for spk, s in chunk_scores.items()}
                best = max(mean_scores, key=mean_scores.get)
                turn_speaker = best if mean_scores[best] >= confidence_threshold else None
            else:
                turn_speaker = None

            records.append((Segment(turn.start, turn.end), track_name, turn_speaker))
        return records

    def _naive() -> List[Tuple[Segment, TrackName, Label]]:
        records: List[Tuple[Segment, TrackName, Label]] = []
        for cs in tqdm(np.arange(0, loaded_audio.duration_seconds, max_chunk_duration)):
            ce = min(cs + max_chunk_duration, loaded_audio.duration_seconds)
            if (ce - cs) >= min_chunk_duration:
                loaded_audio[cs * 1000 : ce * 1000].export(tmp_path, format="wav")
                pred = classifier(str(tmp_path), top_k=1)[0]
                if pred["score"] >= confidence_threshold:
                    records.append((Segment(cs, ce), track_name, pred["label"]))
        return records

    try:
        records = {"diarize": _diarize, "naive": _naive}[mode]()

        merged: List[Tuple[Segment, TrackName, Label]] = []
        cur = None
        for rec in records:
            if cur is None:
                cur = rec
            elif rec[2] == cur[2] and rec[0].start == cur[0].end:
                cur = (Segment(cur[0].start, rec[0].end), track_name, cur[2])
            else:
                merged.append(cur)
                cur = rec
        if cur is not None:
            merged.append(cur)

        return Annotation.from_records(merged)

    finally:
        if tmp_path.exists():
            tmp_path.unlink()