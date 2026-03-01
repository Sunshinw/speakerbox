# speakerbox
--------------------------------------------------------------
# Tang's reimplementation

# Speakerbox — Speaker Identification Fine-Tuner

Fine-tune `wav2vec2` for speaker classification using lazy audio loading.
Works on **Windows (NVIDIA/CUDA)** and **macOS (Apple Silicon/MPS)**.

---

## File Structure

```
speakerbox/
├── main.py            ← model logic (collator, eval, train, apply)
├── RUN_THIS_FILE.py   ← CLI launcher (dataset prep + argument parsing)
└── README.md
```

---

## Quick Start

### Mac (Apple Silicon)
```bash
python RUN_THIS_FILE.py mac \
    --dataset /data/VCTK \
    --output  exps/my_model
```

### Windows (NVIDIA GPU)
```bash
python RUN_THIS_FILE.py windows \
    --dataset C:\data\VCTK \
    --output  exps\my_model
```

---

## All Commands

### `train` — train only
```bash
python RUN_THIS_FILE.py train \
    --dataset PATH \
    --output  exps/my_model
```

### `eval` — evaluate an existing model
```bash
python RUN_THIS_FILE.py eval \
    --dataset    PATH \
    --output     exps/my_model \
    --eval-model exps/my_model/final_speakerbox_epoch_5 \
    --eval-mode  softmax
```

### `train_eval` — train then evaluate in one run
```bash
python RUN_THIS_FILE.py train_eval \
    --dataset PATH \
    --output  exps/my_model
```

### `mac` — preset shortcut for Apple Silicon
```bash
python RUN_THIS_FILE.py mac \
    --dataset /data/VCTK \
    --output  exps/my_model
```
Auto-sets: `--audio-backend soundfile`, `--eval-mode softmax`, `fp16=False`

### `windows` — preset shortcut for Windows/NVIDIA
```bash
python RUN_THIS_FILE.py windows \
    --dataset C:\data\VCTK \
    --output  exps\my_model
```
Auto-sets: `--audio-backend librosa`, `--eval-mode pipeline`, `--fp16`

---

## Full Argument Reference

| Flag | Commands | Default | Description |
|---|---|---|---|
| `--dataset PATH` | all | *(required)* | Root folder of `.wav`/`.flac` audio files |
| `--output PATH` | all | `exps/model` | Where to save the trained model |
| `--audio-backend` | all | `soundfile` | `soundfile` (fast, 16 kHz) or `librosa` (auto-resamples) |
| `--min-speaker-files N` | all | `1` | Min unique files per speaker to include |
| `--model-base HF_ID` | train, train_eval | `superb/wav2vec2-base-superb-sid` | HuggingFace base model to fine-tune |
| `--max-duration SEC` | train, train_eval | `3.0` | Clip length in seconds (truncated/padded) |
| `--epochs N` | train, train_eval | `10` | Number of training epochs |
| `--batch N` | train, train_eval | `4` | Per-device train batch size |
| `--accum N` | train, train_eval | `4` | Gradient accumulation steps (effective batch = batch × accum) |
| `--lr LR` | train, train_eval | `3e-5` | Learning rate |
| `--fp16` | train, train_eval | off | Enable mixed-precision (NVIDIA only) |
| `--save-steps N` | train, train_eval | `200` | Save checkpoint every N steps |
| `--metadata-cache PATH` | train, train_eval | auto | Cache path for Audio-cast dataset. `''` disables caching |
| `--seed N` | train, train_eval | `42` | Global random seed |
| `--resume` | train, train_eval | on | Resume from latest checkpoint if found |
| `--no-resume` | train, train_eval | — | Force training from scratch |
| `--eval-model PATH` | eval, train_eval | `--output` | Model path to evaluate |
| `--eval-mode` | eval, train_eval | `softmax` | `softmax` (batched, mac-friendly) or `pipeline` (HF pipeline, windows-style) |

---

## Output Structure

After training, `--output` contains:

```
exps/my_model/
├── checkpoint-200/                  ← HF Trainer checkpoints (auto-saved every --save-steps)
├── checkpoint-400/
├── final_speakerbox_epoch_1/        ← Full model snapshot saved by EpochArchiveCallback
├── final_speakerbox_epoch_2/
├── speakerboxdts_metadata_ds/       ← Cached audio-path dataset (skip re-casting on rerun)
├── config.json                      ← Final model config
├── pytorch_model.bin                ← Final model weights
└── results.md                       ← Eval metrics (accuracy, precision, recall, loss)
```

---

## Code Architecture

### `RUN_THIS_FILE.py`

| Line | Name | Topic |
|---|---|---|
| 1 | *(module docstring)* | Embedded usage examples |
| 35–44 | imports | argparse, pandas, HF datasets, local `main.py` import via `sys.path` |
| 54 | `def prepare_dataset` | Scans audio folder, derives labels from filename, builds 80/10/10 HF DatasetDict |
| 76 | *(inside prepare_dataset)* | Shuffle + speaker filter by `min_speaker_files` |
| 83–96 | *(inside prepare_dataset)* | Train/valid/test split, cast to `Audio(decode=False)`, class-encode labels |
| 103 | `def cmd_train` | Builds `trainer_args` dict, calls `train()` from `main.py` |
| 137 | `def cmd_eval` | Casts test split, calls `eval_model()` from `main.py`, logs metrics |
| 160 | `def build_parser` | Defines all argparse sub-commands and flags |
| 168–170 | *(inside build_parser)* | Registers `train`, `eval`, `train_eval` sub-commands |
| 172–193 | *(inside build_parser)* | Shared flags: `--dataset`, `--output`, `--audio-backend`, `--min-speaker-files` |
| 195–250 | *(inside build_parser)* | Train flags: `--epochs`, `--batch`, `--accum`, `--lr`, `--fp16`, `--save-steps`, `--metadata-cache`, `--seed`, `--resume`/`--no-resume` |
| 252–265 | *(inside build_parser)* | Eval flags: `--eval-model`, `--eval-mode` |
| 267–288 | *(inside build_parser)* | `mac` shortcut sub-command with Apple Silicon presets |
| 290–311 | *(inside build_parser)* | `windows` shortcut sub-command with NVIDIA presets |
| 320 | `def main` | Dispatches parsed args to `cmd_train`, `cmd_eval`, or both |
| 324–332 | *(inside main)* | `mac` branch — injects preset values then runs train + eval |
| 334–342 | *(inside main)* | `windows` branch — injects preset values then runs train + eval |

---

### `main.py`

| Line | Name | Topic |
|---|---|---|
| 1–16 | imports | Standard lib, numpy, torch, transformers; TYPE_CHECKING-only HF/pyannote imports |
| 22–31 | `EVAL_RESULTS_TEMPLATE` | Markdown template string written to `results.md` after eval |
| 35 | `class LazyAudioCollator` | Loads audio from disk and extracts features **per batch** — avoids pre-materialising the whole dataset into RAM |
| 49 | `def __init__` | Stores feature extractor, computes `max_len` in samples, stores `audio_backend` |
| 54 | `def _load` | Dispatches audio loading to `soundfile.read` or `librosa.load` based on `audio_backend` |
| 63 | `def __call__` | Called by HF Trainer each batch — resolves paths, loads waveforms, runs feature extractor, returns `input_values` + `labels` tensors |
| 88 | `def preprocess_function` | Batch-maps audio → `input_values` for eval pre-caching; supports both audio backends |
| 126 | `class EpochArchiveCallback` | HF TrainerCallback that snapshots a complete model at the end of every epoch |
| 134 | `def __init__` | Stores validation dataset, feature extractor, `run_eval`, `eval_mode`, `audio_backend` |
| 148 | `def on_epoch_end` | Finds latest `checkpoint-*`, copies to `final_speakerbox_epoch_N`, saves feature extractor, optionally runs `eval_model` |
| 178 | `def eval_model` | Evaluates a saved model; returns `(accuracy, precision, recall, loss)` and writes `results.md` |
| 208–242 | *(pipeline branch)* | `eval_mode="pipeline"` — HF pipeline + librosa, one sample at a time; original Windows-style path |
| 219 | `def _predict_one` | Inner function for pipeline eval — loads audio, runs classifier, aligns scores to label order |
| 244–301 | *(softmax branch)* | `eval_mode="softmax"` — loads model directly, batched forward pass + `torch.softmax`; mac/MPS-friendly |
| 268 | `def _predict_batch` | Inner function for softmax eval — loads batch of wavs, runs model, returns pred labels + softmax scores |
| 303–321 | *(shared metrics)* | `accuracy_score`, `precision_score`, `recall_score`, `log_loss` computed once regardless of eval mode |
| 327 | `def _find_last_checkpoint` | Globs `checkpoint-*` in output dir, sorts by step number, returns latest path or `None` — prevents crash when no checkpoint exists |
| 347 | `def train` | Main training entry point — device setup, model init, metadata caching, HF Trainer construction and `.train()` call |
| 394–408 | *(inside train)* | Device selection: CUDA → MPS → CPU with automatic fp16/no_cuda adjustment |
| 411–416 | *(inside train)* | Feature extractor load + `label2id`/`id2label` lookup tables from dataset |
| 418–428 | *(inside train)* | `Wav2Vec2ForSequenceClassification` loaded from `model_base`, moved to device |
| 430–450 | *(inside train)* | Metadata cache logic — load from disk if exists, otherwise cast + save to disk |
| 452–453 | *(inside train)* | `TrainingArguments` built from `trainer_arguments_kws` dict |
| 455–463 | *(inside train)* | `compute_metrics` — argmax logits → accuracy for HF Trainer eval loop |
| 465–471 | *(inside train)* | `LazyAudioCollator` and `EpochArchiveCallback` instantiated |
| 473–481 | *(inside train)* | `Trainer` assembled with model, args, train dataset, collator, metrics, callback |
| 484–485 | *(inside train)* | `_find_last_checkpoint` called; result passed to `trainer.train()` — `None` = fresh start, path = resume |
| 494 | `def apply` | Runs trained model over a full audio file in sliding chunks; returns a `pyannote.core.Annotation` |
| 540 | `def _diarize` | Inner function — diarizes first with pyannote, then classifies each speaker turn |
| 571 | `def _naive` | Inner function — slides a fixed window, classifies each chunk independently |
| 582–603 | *(inside apply)* | Merges touching segments with same label; cleans up temp wav file in `finally` block |


--------------------------------------------------------------
[![Build Status](https://github.com/CouncilDataProject/speakerbox/workflows/CI/badge.svg)](https://github.com/CouncilDataProject/speakerbox/actions)
[![Documentation](https://github.com/CouncilDataProject/speakerbox/workflows/Documentation/badge.svg)](https://CouncilDataProject.github.io/speakerbox)
[![status](https://joss.theoj.org/papers/49cfcef1769c812ce4ff2e388a5c7641/status.svg)](https://joss.theoj.org/papers/49cfcef1769c812ce4ff2e388a5c7641)

Few-Shot Multi-Recording Speaker Identification Transformer Fine-Tuning and Application

---

## Installation

**Stable Release:** `pip install speakerbox`<br>
**Development Head:** `pip install git+https://github.com/CouncilDataProject/speakerbox.git`

## Documentation

For full package documentation please visit [councildataproject.github.io/speakerbox](https://councildataproject.github.io/speakerbox).

## Example Usage Video

[![screenshot from example usage youtube video](https://raw.githubusercontent.com/CouncilDataProject/speakerbox/main/docs/_static/images/speakerbox-example-video-screenshot.png)](https://youtu.be/SK2oVqSKPTE)

Link: [https://youtu.be/SK2oVqSKPTE](https://youtu.be/SK2oVqSKPTE)

In the example video, we use the Speakerbox library to quickly annotate a 
dataset of audio clips from the show 
[The West Wing](https://en.wikipedia.org/wiki/The_West_Wing) 
and train a speaker identification model to identify three of 
the show's characters (President Bartlet, Charlie Young, and Leo McGarry).

## Problem

Given a set of recordings of multi-speaker recordings:

```
example/
├── 0.wav
├── 1.wav
├── 2.wav
├── 3.wav
├── 4.wav
└── 5.wav
```

Where each recording has some or all of a set of speakers, for example:

-   0.wav -- contains speakers: A, B, C
-   1.wav -- contains speakers: A, C
-   2.wav -- contains speakers: B, C
-   3.wav -- contains speakers: A, B, C
-   4.wav -- contains speakers: A, B, C
-   5.wav -- contains speakers: A, B, C

You want to train a model to classify portions of audio as one of the N known speakers
in future recordings not included in your original training set.

`f(audio) -> [(start_time, end_time, speaker), (start_time, end_time, speaker), ...]`

i.e. `f(audio) -> [(2.4, 10.5, "A"), (10.8, 14.1, "D"), (14.8, 22.7, "B"), ...]`

The `speakerbox` library contains methods for both generating datasets for annotation
and for utilizing multiple audio annotation schemes to train such a model.

![Typical workflow to prepare a speaker identification dataset and fine-tune a new model using tools provided from the Speakerbox library. The user starts with a collection of audio files that include portions speech from the speakers they want to train a model to identify. The `diarize_and_split_audio` function will create a new directory with the same name as the audio file, diarize the audio file, and finally, sort the audio portions produced from diarization into sub-directories within this new directory. The user should then manually rename each of the produced sub-directories to the correct speaker identifier (i.e. the speaker's name or a unique id) and additionally remove any incorrectly diarized or mislabeled portions of audio. Finally, the user can prepare training, evaluation, and testing datasets (via the `expand_labeled_diarized_audio_dir_to_dataset` and `preprocess_dataset` functions) and fine-tune a new speaker identification model (via the `train` function).](https://raw.githubusercontent.com/CouncilDataProject/speakerbox/main/docs/_static/images/workflow.png)

The following table shows model performance results as the dataset size increases:

| dataset_size   | mean_accuracy   | mean_precision   | mean_recall   | mean_training_duration_seconds   |
|:---------------|----------------:|-----------------:|--------------:|---------------------------------:|
| 15-minutes     | 0.874 ± 0.029   | 0.881 ± 0.037    | 0.874 ± 0.029 | 101 ± 1                          |
| 30-minutes     | 0.929 ± 0.006   | 0.94 ± 0.007     | 0.929 ± 0.006 | 186 ± 3                          |
| 60-minutes     | 0.937 ± 0.02    | 0.94 ± 0.017     | 0.937 ± 0.02  | 453 ± 7                          |

All results reported are the average of five model training and evaluation trials for each
of the different dataset sizes. All models were fine-tuned using an NVIDIA GTX 1070 TI.

**Note:** this table can be reproduced in ~1 hour using an NVIDIA GTX 1070 TI by:

Installing the example data download dependency:

```bash
pip install speakerbox[example_data]
```

Then running the following commands in Python:

```python
from speakerbox.examples import (
    download_preprocessed_example_data,
    train_and_eval_all_example_models,
)

# Download and unpack the preprocessed example data
dataset = download_preprocessed_example_data()

# Train and eval models with different subsets of the data
results = train_and_eval_all_example_models(dataset)
```

## Workflow

### Diarization

We quickly generate an annotated dataset by first diarizing (or clustering based
on the features of speaker audio) portions of larger audio files and splitting each the
of the clusters into their own directories that you can then manually clean up
(by removing incorrectly clustered audio segments).

#### Notes

-   It is recommended to have each larger audio file named with a unique id that
    can be used to act as a "recording id".
-   Diarization time depends on machine resources and make take a long time -- one
    potential recommendation is to run a diarization script overnight and clean up the
    produced annotations the following day.
-   During this process audio will be duplicated in the form of smaller audio clips --
    ensure you have enough space on your machine to complete this process before
    you begin.
-   Clustering accuracy depends on how many speakers there are, how distinct their
    voices are, and how much speech is talking over one-another.
-   If possible, try to find recordings where speakers have a roughly uniform distribution
    of speaking durations.

⚠️ To use the diarization portions of `speakerbox` you need to complete the
following steps: ⚠️

1. Visit [hf.co/pyannote/speaker-diarization](https://hf.co/pyannote/speaker-diarization)
   and accept user conditions.
2. Visit [hf.co/pyannote/segmentation](https://hf.co/pyannote/segmentation)
   and accept user conditions.
3. Visit [hf.co/settings/tokens](https://hf.co/settings/tokens) to create an access token
   (only if you had to complete 1.)

**Diarize a single file:**

```python
from speakerbox import preprocess

# The token can also be provided via the 'HUGGINGFACE_TOKEN` environment variable.
diarized_and_split_audio_dir = preprocess.diarize_and_split_audio(
    "0.wav",
    hf_token="token-from-hugging-face",
)
```

**Diarize all files in a directory:**
```python
from speakerbox import preprocess
from pathlib import Path
from tqdm import tqdm

# Iterate over all 'wav' format files in a directory called 'data'
for audio_file in tqdm(list(Path("data").glob("*.wav"))):
    # The token can also be provided via the 'HUGGINGFACE_TOKEN` environment variable.
    diarized_and_split_audio_dir = preprocess.diarize_and_split_audio(
        audio_file,
        # Create a new directory to place all created sub-directories within
        storage_dir=f"diarized-audio/{audio_file.stem}",
        hf_token="token-from-hugging-face",
    )
```

### Cleaning

Diarization will produce a directory structure organized by unlabeled speakers with
the audio clips that were clustered together.

For example, if `"0.wav"` had three speakers, the produced directory structure may look
like the following tree:

```
0/
├── SPEAKER_00
│   ├── 567-12928.wav
│   ├── ...
│   └── 76192-82901.wav
├── SPEAKER_01
│   ├── 34123-38918.wav
│   ├── ...
│   └── 88212-89111.wav
└── SPEAKER_02
    ├── ...
    └── 53998-62821.wav
```

We leave it to you as a user to then go through these directories and remove any audio
clips that were incorrectly clustered together as well as renaming the sub-directories
to their correct speaker labels. For example, labelled sub-directories may look like
the following tree:

```
0/
├── A
│   ├── 567-12928.wav
│   ├── ...
│   └── 76192-82901.wav
├── B
│   ├── 34123-38918.wav
│   ├── ...
│   └── 88212-89111.wav
└── D
    ├── ...
    └── 53998-62821.wav
```

#### Notes

-   Most operating systems have an audio playback application to queue an entire directory
    of audio files as a playlist for playback. This makes it easy to listen to a whole
    unlabeled sub-directory (i.e. "SPEAKER_00") at a time and pause playback and remove
    files from the directory which were incorrectly clustered.
-   If any clips have overlapping speakers, it is up to you as a user if you want to
    remove those clips or keep them and properly label them with the speaker you wish to
    associate them with.

### Training Preparation

Once you have annotated what you think is enough recordings, you can try preparing
a dataset for training.

The following functions will prepare the audio for training by:

1. Finding all labeled audio clips in the provided directories
2. Chunk all found audio clips into smaller duration clips _(parametrizable)_
3. Check that the provided annotated dataset meets the following conditions:
    1. There is enough data such that the training, test, and validation subsets all
       contain different recording ids.
    2. There is enough data such that the training, test, and validation subsets each
       contain all labels present in the whole dataset.

#### Notes

-   During this process audio will be duplicated in the form of smaller audio clips --
    ensure you have enough space on your machine to complete this process before
    you begin.
-   Directory names are used as recording ids during dataset construction.

```python
from speakerbox import preprocess

dataset = preprocess.expand_labeled_diarized_audio_dir_to_dataset(
    labeled_diarized_audio_dir=[
        "0/",  # The cleaned and checked audio clips for recording id 0
        "1/",  # ... recording id 1
        "2/",  # ... recording id 2
        "3/",  # ... recording id 3
        "4/",  # ... recording id 4
        "5/",  # ... recording id 5
    ]
)

dataset_dict, value_counts = preprocess.prepare_dataset(
    dataset,
    # good if you have large variation in number of data points for each label
    equalize_data_within_splits=True,
    # set seed to get a reproducible data split
    seed=60,
)

# You can print the value_counts dataframe to see how many audio clips of each label
# (speaker) are present in each data subset.
value_counts
```

### Model Training and Evaluation

Once you have your dataset prepared and available, you can provide it directly to the
training function to begin training a new model.

The `eval_model` function will store a filed called `results.md` with the accuracy,
precision, and recall of the model and additionally store a file called
`validation-confusion.png` which is a
[confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix).

#### Notes

-   The model (and evaluation metrics) will be stored in a new directory called
    `trained-speakerbox` _(parametrizable)_.
-   Training time depends on how much data you have annotated and provided.
-   It is recommended to train with an NVidia GPU with CUDA available to speed up
    the training process.
-   Speakerbox has only been tested on English-language audio and the base model for
    fine-tuning was trained on English-language audio. We provide no guarantees as to
    it's effectiveness on non-English-language audio. If you try Speakerbox on with
    non-English-language audio, please let us know!

```python
from speakerbox import train, eval_model

# dataset_dict comes from previous preparation step
train(dataset_dict)

eval_model(dataset_dict["valid"])
```

## Model Inference

Once you have a trained model, you can use it against a new audio file:

```python
from speakerbox import apply

annotation = apply(
    "new-audio.wav",
    "path-to-my-model-directory/",
)
```

The apply function returns a
[pyannote.core.Annotation](http://pyannote.github.io/pyannote-core/structure.html#annotation).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.

## Citation

```bibtex
@article{Brown2023,
    doi = {10.21105/joss.05132},
    url = {https://doi.org/10.21105/joss.05132},
    year = {2023},
    publisher = {The Open Journal},
    volume = {8},
    number = {83},
    pages = {5132},
    author = {Eva Maxfield Brown and To Huynh and Nicholas Weber},
    title = {Speakerbox: Few-Shot Learning for Speaker Identification with Transformers},
    journal = {Journal of Open Source Software}
} 
```

**MIT License**
