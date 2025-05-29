import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from evaluation.const import LEN

from scipy.stats import pearsonr
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

MODEL_NAME = "ai-forever/ruT5-large"
DATASET_NAME = "hgissbkh/seahorse"
LANGUAGES = ["ru", "en"]
QUESTION = "conciseness"
MAX_INPUT_LENGTH = 2048 
MAX_TARGET_LENGTH = 2
OUTPUT_DIR = "./seahorse_metric" 
FORMAT = "текст:\n {} саммари:\n {}"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

import random
import numpy as np
import torch

SEED = 42

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

seahorse = load_dataset(DATASET_NAME)

def filter_data(example):
    if(example[QUESTION] == 0.5):
        return False

    example[QUESTION] = int(example[QUESTION])
    return example['lang'] in LANGUAGES and len(example['summary']) > LEN

seahorse_filtered = seahorse.filter(filter_data)

def filter_long_examples(example):
    inputs = FORMAT.format(example['text'], example['summary'])
    tokenized = tokenizer(inputs, truncation=False)
    return len(tokenized['input_ids']) <= MAX_INPUT_LENGTH

seahorse_filtered = seahorse_filtered.filter(filter_long_examples)

train_dataset = seahorse_filtered['train']
validation_dataset = seahorse_filtered['validation']

zero_token_str = '▁0'
one_token_str = '▁1'

tokenized_0 = tokenizer(zero_token_str).input_ids 
tokenized_1 = tokenizer(one_token_str).input_ids 

print(f"Tokenizing '{zero_token_str}': {tokenized_0} -> {tokenizer.convert_ids_to_tokens(tokenized_0)}")
print(f"Tokenizing '{one_token_str}': {tokenized_1} -> {tokenizer.convert_ids_to_tokens(tokenized_1)}")

zero_token_id = tokenized_0[0]
one_token_id = tokenized_1[0]
eos_token_id = tokenizer.eos_token_id
MAX_TARGET_LENGTH = len(tokenized_0) # Should be 2
print(f"Max target length set to: {MAX_TARGET_LENGTH}")

labels_map = {
    0: zero_token_str,
    1: one_token_str
}

def preprocess_function(examples):
    inputs = [FORMAT.format(article, summary)
              for article, summary in zip(examples['text'], examples['summary'])]
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False, # Defer padding to DataCollator
    )

    target_texts = [labels_map[label] for label in examples[QUESTION]]
    labels = tokenizer(
        target_texts,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False, # Defer padding to DataCollator
    ).input_ids

    model_inputs["labels"] = labels

    return model_inputs


validation_ru = validation_dataset.filter(lambda x: x['lang'] == 'ru')

train_tokenized = train_dataset.map(preprocess_function, batched=True, num_proc=16, remove_columns=train_dataset.column_names)
validation_tokenized = validation_ru.map(preprocess_function, batched=True, num_proc=16, remove_columns=validation_dataset.column_names)

print(f"Train: {len(train_tokenized)}, Validation: {len(validation_tokenized)}")

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, dropout_rate=0.11)
# model.gradient_checkpointing_enable()

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    # Logits shape: (batch_size, sequence_length, vocab_size)
    # Labels shape: (batch_size, sequence_length)
    logits = predictions[0]

    first_token_logits = logits[:, 0, :]
    labels_binary = np.where(labels[:, 0] == one_token_id, 1, 0)

    logits_for_0 = first_token_logits[:, zero_token_id]
    logits_for_1 = first_token_logits[:, one_token_id]    
    scores_diff = logits_for_1 - logits_for_0

    pearson_corr, _ = pearsonr(scores_diff, labels_binary)
    auc = roc_auc_score(labels_binary, scores_diff)

    logits_0_1 = first_token_logits[:, [zero_token_id, one_token_id]]
    probabilities = softmax(logits_0_1, axis=1)

    threshold = 0.5
    predictions = (probabilities[:, 1] > threshold).astype(int)
    accuracy = accuracy_score(labels_binary, predictions)
    f1 = f1_score(labels_binary, predictions, average="weighted")
    threshold_predictions = np.where(probabilities[:, 1] > threshold, 1, 0)
    
    confidences = np.max(probabilities, axis=1)
    mean_confidence_overall = np.mean(confidences)

    correct_predictions_mask = (threshold_predictions == labels_binary)
    incorrect_predictions_mask = ~correct_predictions_mask

    mean_confidence_correct = np.mean(confidences[correct_predictions_mask]) if np.any(correct_predictions_mask) else 0.0
    mean_confidence_incorrect = np.mean(confidences[incorrect_predictions_mask]) if np.any(incorrect_predictions_mask) else 0.0

    predicted_token_ids_overall = np.argmax(first_token_logits, axis=-1)
    is_valid_target_token_pred = np.logical_or(predicted_token_ids_overall == zero_token_id,
                                               predicted_token_ids_overall == one_token_id)
    invalid_prediction_rate = 1.0 - np.mean(is_valid_target_token_pred)

    return {
        "auc": auc,
        "accuracy": accuracy,
        "f1": f1, 
        "pearson": pearson_corr,
        "invalid_target_rate": invalid_prediction_rate, # Доля предсказаний не '▁0' и не '▁1'
        "mean_confidence_overall": mean_confidence_overall,
        "mean_confidence_correct": mean_confidence_correct,
        "mean_confidence_incorrect": mean_confidence_incorrect,
    }
    
    

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=4,
    learning_rate=2.5e-5,
    lr_scheduler_type="cosine",
    # lr_scheduler_kwargs={"num_cycles":2},
    per_device_train_batch_size=4,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,
    weight_decay=0.04, 
    label_smoothing_factor=0.2,
    # max_grad_norm=1.0,
    optim="adafactor",
    warmup_ratio=0.1,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_strategy="steps",
    logging_steps=200,
    save_total_limit=2,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    load_best_model_at_end=True,
    metric_for_best_model="eval_auc", 
    greater_is_better=True,
    logging_dir=f"./{OUTPUT_DIR}/logs",
    report_to="tensorboard",
    seed=42,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=validation_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator, # Use the data collator
    compute_metrics=compute_metrics,
)

print("Starting training...")
train_result = trainer.train()
print(train_result)

print("Evaluating on validation set...")
eval_results_trainer = trainer.evaluate()
print(eval_results_trainer)

trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
