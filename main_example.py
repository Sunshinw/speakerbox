from speakerbox.examples import (
    download_preprocessed_example_data,
    train_and_eval_all_example_models,
)

# Download and unpack the preprocessed example data
dataset = download_preprocessed_example_data()

# Train and eval models with different subsets of the data
results = train_and_eval_all_example_models(dataset)