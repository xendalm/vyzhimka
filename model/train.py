import os
import logging
import gc
import numpy as np
import torch
import nltk
from common.const import (
    TASK_PROMPT,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    SEED
)
from common import utils
from datasets import (
    DatasetDict,
    load_from_disk
)
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
import evaluate

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_NAME = "ai-forever/FRED-T5-large"
OUTPUT_DIR = "fred-t5_summarization_combined"
TRAINING_DATA = "./data/combined_data"

NUM_EPOCHS = 6
TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 6
EVAL_BATCH_SIZE = 64
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.015
SAVE_TOTAL_LIMIT = 2
REPORT_TO = "tensorboard"
WARMUP_RATIO = 0.1
SCHEDULER_TYPE = "linear"

use_subset = False
subset_train_size = 10000
subset_val_size = 1000

def preprocess(examples, tokenizer):
    inputs = tokenizer(
        [TASK_PROMPT + doc for doc in examples["text"]],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False, # Defer padding to DataCollator
    )
    targets = tokenizer(
        examples["summary"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False, # Defer padding to DataCollator
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


rouge_metric = evaluate.load("rouge")
def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    # Replace -100 in the labels as we can't decode them.
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {key: value for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 3) for k, v in result.items()}


if __name__ == "__main__":
    utils.set_seed(SEED)

    data = load_from_disk(TRAINING_DATA)
    datasets = data.train_test_split(test_size=0.05, seed=42, shuffle=True)
    datasets["validation"] = datasets.pop("test")

    logger.info(f"Datasets: {datasets}")

    num_proc = os.cpu_count() // 2

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_datasets = datasets.map(preprocess, batched=True, fn_kwargs={"tokenizer": tokenizer},
                             remove_columns=datasets["train"].column_names, num_proc=num_proc)
    
    logger.info(f"Tokenized datasets: {tokenized_datasets}")
    
    gc.collect()
    torch.cuda.empty_cache()

    if use_subset:
        train_data = tokenized_datasets["train"].shuffle(seed=SEED).select(range(min(subset_train_size, len(tokenized_datasets["train"]))))
        val_data = tokenized_datasets["validation"].shuffle(seed=SEED).select(range(min(subset_val_size, len(tokenized_datasets["validation"]))))
        final_datasets = DatasetDict({"train": train_data, "validation": val_data})
    else:
        final_datasets = tokenized_datasets

    logger.info(f"Final datasets: {final_datasets}")

    total_train_samples = len(final_datasets['train'])
    effective_batch_size = TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    steps_per_epoch = (total_train_samples + effective_batch_size - 1) // effective_batch_size
    total_training_steps = steps_per_epoch * NUM_EPOCHS

    eval_steps = max(100, steps_per_epoch // 10)
    save_steps = eval_steps * 2 # Сохраняем реже, чем оцениваем
    logging_steps = max(50, eval_steps // 10) # Логируем чаще

    logger.info(f"Effective batch size: {effective_batch_size}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Total training steps: {total_training_steps}")
    logger.info(f"Logging steps: {logging_steps}")
    logger.info(f"Eval steps: {eval_steps}")
    logger.info(f"Save steps: {save_steps}")


    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim="adafactor",
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=SCHEDULER_TYPE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        logging_strategy="steps",
        logging_steps=logging_steps,
        save_total_limit=SAVE_TOTAL_LIMIT,
        bf16=torch.cuda.is_bf16_supported(), 
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=REPORT_TO,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss", 
        greater_is_better=False,
        save_safetensors=False,
        logging_dir=f"{OUTPUT_DIR}/logs",
        # gradient_checkpointing=True,
        seed=SEED,
        data_seed=SEED,
    )


    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=final_datasets["train"],
        eval_dataset=final_datasets["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    logger.info("Starting training...")
    # train_result = trainer.train(resume_from_checkpoint="")
    train_result = trainer.train()
    logger.info("Saving model...")
    trainer.save_model()
    logger.info("Training complete.")

    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info(f"Metrics: {metrics}")

    logger.info("Starting evaluation...")
    eval_result = trainer.evaluate()
    trainer.log_metrics("eval", eval_result)
    trainer.save_metrics("eval", eval_result)
    logger.info(f"Metrics: {eval_result}")

    del model, trainer, data_collator
    gc.collect()
    torch.cuda.empty_cache()
