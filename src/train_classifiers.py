import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import os
import evaluate
import sentencepiece
import argparse

model_saving_dir = os.path.join("saved_models")
if not os.path.isdir(model_saving_dir):
    os.mkdir(model_saving_dir)

accuracy_metric = evaluate.load("accuracy")


def prepare_data(dataset_file, prompt_style, generation_model, dataset_type, split):

    df = pd.read_csv(dataset_file, index_col=0)
    df = df.dropna(ignore_index=True)

    df = df.loc[((df.source == dataset_type) & (df.model == "human") & (df.split == split)) | ((df.source == dataset_type) & (df.prompt_style == prompt_style) & (df.model == generation_model) & (df.split == split))]
    texts = df["text"].to_list()
    labels = df["label"].to_list()

    return texts, labels

class DetectionDataset(Dataset):

    def __init__(self, encodings, labels): #tokenized encodings and labels
        self.encodings=encodings
        self.labels=labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset_type", 
                        type=str, 
                        required=True, 
                        help='the type of dataset: "abstracts", "reviews", "news", "qa".')
    parser.add_argument("--prompt_type", 
                        type=str, 
                        required=True, 
                        help='the prompt subtype used for generating text. one from "zero_shot_baseline", "chain_of_thought_zero_shot", "style_information_style_example", "chain_of_thought_one_shot", "in_context_learning_3-shot", "self-refine"')
    parser.add_argument('--generation_model',
                        type=str, 
                        required=True,
                        help='model used for generating text. One from "llama3.3", "qwen14b", "qwen72b", "qwen32b", "mistral", solar", "deepseek"')
    parser.add_argument('--detection_model',
                        type=str,
                        required=True,
                        help='model used for classification: one from "xlm-roberta-base", "roberta-base", "microsoft/deberta-v3-base", "google-bert/bert-base-multilingual-cased", "openai-community/roberta-large-openai-detector", "google/electra-small-discriminator"/"google/electra-large-discriminator"/"google/electra-base-discriminator"')

    
    args = parser.parse_args()
    if args.prompt_type == "self-refine":
        dataset_file = os.path.join("full_dataset", "dataset_multistage_EN.csv") # the csv/json file with the entire dataset
    
    else:
        dataset_file = os.path.join("full_dataset", "dataset_full_EN.csv")


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.detection_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.detection_model, num_labels=2)
    model.to(device)

    model_path = os.path.join(model_saving_dir, args.detection_model + "_" + args.generation_model + "_" + args.dataset_type + "_" + args.prompt_type)
    
    X_train, y_train = prepare_data(dataset_file=dataset_file,
                                  prompt_style=args.prompt_type,
                                  generation_model=args.generation_model,
                                  dataset_type=args.dataset_type,
                                  split="train")
    
    X_val, y_val = prepare_data(dataset_file=dataset_file,
                                  prompt_style=args.prompt_type,
                                  generation_model=args.generation_model,
                                  dataset_type=args.dataset_type,
                                  split="val")
    
    X_test, y_test = prepare_data(dataset_file=dataset_file,
                                  prompt_style=args.prompt_type,
                                  generation_model=args.generation_model,
                                  dataset_type=args.dataset_type,
                                  split="test")
    

    train_encodings = tokenizer(X_train, truncation=True, padding=True, return_tensors="pt", max_length=512)
    val_encodings = tokenizer(X_val, truncation=True, padding=True, return_tensors="pt", max_length=512)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, return_tensors="pt", max_length=512)

    train_dataset = DetectionDataset(encodings=train_encodings, labels=y_train)
    val_dataset = DetectionDataset(encodings=val_encodings, labels=y_val)
    test_dataset = DetectionDataset(encodings=test_encodings, labels=y_test)

    training_args = TrainingArguments(seed=42,
                                    output_dir=model_path,
                                    overwrite_output_dir=True,
                                    learning_rate=2e-5,
                                    num_train_epochs=3,
                                    weight_decay=0.01,
                                    per_device_train_batch_size=16,
                                    per_device_eval_batch_size=16,
                                    eval_strategy="steps",
                                    save_strategy="steps",
                                    logging_strategy="steps",
                                    load_best_model_at_end=True,
                                    metric_for_best_model="accuracy",
                                    logging_steps=20,
                                    eval_steps=20,
                                    save_steps=200,
                                    report_to="none",
                                    push_to_hub=False)


    trainer = Trainer(model=model,
                  args=training_args,
                  train_dataset=train_dataset,
                  eval_dataset=val_dataset,
                  compute_metrics=compute_metrics,
                  callbacks=[EarlyStoppingCallback(10)])
    
    
    trainer.train()
    trainer.save_model(model_path)

    predictions_all = trainer.predict(test_dataset)
    prediction_values = predictions_all.predictions
    prediction_metrics = predictions_all.metrics

    trainer.save_metrics("test", prediction_metrics, combined=False)

    trainer_history_all = trainer.state.log_history 
    trainer_history_metrics = trainer_history_all[:-1] 
    trainer_history_training_time = trainer_history_all[-1]

    trainer_history_training_set = []
    trainer_history_eval_set = []

    # Loop through metrics and filter for training and eval metrics
    for item in trainer_history_metrics:
        item_keys = list(item.keys())
        # Check to see if "eval" is in the keys of the item
        if any("eval" in item for item in item_keys):
            trainer_history_eval_set.append(item)
        else:
            trainer_history_training_set.append(item)

    trainer_history_training_df = pd.DataFrame(trainer_history_training_set)
    trainer_history_eval_df = pd.DataFrame(trainer_history_eval_set)

    plt.figure(figsize=(10, 6))
    plt.plot(trainer_history_training_df["step"], trainer_history_training_df["loss"], label="Training loss")
    plt.plot(trainer_history_eval_df["step"], trainer_history_eval_df["eval_loss"], label="Evaluation loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Training and evaluation loss over time")
    plt.legend()
    plt.savefig(f"plots/{args.detection_model}_{args.generation_model}_{args.dataset_type}_{args.prompt_type}.png")