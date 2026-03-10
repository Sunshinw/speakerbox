#!/bin/bash

# Run the training script
python RUN_THIS_FILE.py mac \
  --dataset "/Users/pakap/Documents/Senior/Code/Dataset/libri500_WAV" \
  --output "exps/libri500_2times" \
  --train-mode "scratch" \