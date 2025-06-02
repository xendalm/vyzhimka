import os
import random
import logging
import gc
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from trl import DPOTrainer, DPOConfig
from common import utils
from common.const import (
    TASK_PROMPT,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    SEED
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

BASE_MODEL_PATH = "fred-t5_summarization_synth_2"
REF_MODEL_PATH = "fred-t5_summarization_synth_2"

PREFERENCE_DATA_FILE = "data/preference_data_conciseness.jsonl"

OUTPUT_DIR_DPO = "fred-t5_summarization_dpo"

DPO_BETA = 0.1 # DPO beta parameter (controls the strength of the preference signal)

NUM_EPOCHS_DPO = 3 # DPO usually requires fewer epochs
TRAIN_BATCH_SIZE_DPO = 4
GRADIENT_ACCUMULATION_STEPS_DPO = 8
LEARNING_RATE_DPO = 1e-5
WEIGHT_DECAY_DPO = 0.0
WARMUP_RATIO_DPO = 0.05
SAVE_TOTAL_LIMIT_DPO = 2
REPORT_TO_DPO = "tensorboard"
SCHEDULER_TYPE = "linear"

# Evaluation during DPO training is optional and can be slow.
# We will rely on evaluating the final model with eval.py instead.
# EVAL_BATCH_SIZE_DPO = 32
# EVAL_STEPS_DPO = 500 # Set to None if no eval during training

def preprocess_for_dpo(examples, tokenizer):
    prompt_tokenized = tokenizer(
        [TASK_PROMPT + text for text in examples["text"]],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False, # Defer padding
    )

    chosen_tokenized = tokenizer(
        examples["chosen"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False, # Defer padding
    )

    rejected_tokenized = tokenizer(
        examples["rejected"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False, # Defer padding
    )

    return {
        "prompt_input_ids": prompt_tokenized["input_ids"],
        "prompt_attention_mask": prompt_tokenized["attention_mask"],
        "chosen_input_ids": chosen_tokenized["input_ids"],
        "chosen_attention_mask": chosen_tokenized["attention_mask"],
        "rejected_input_ids": rejected_tokenized["input_ids"],
        "rejected_attention_mask": rejected_tokenized["attention_mask"],
    }


if __name__ == "__main__":
    utils.set_seed(SEED)

    try:
        dataset = load_dataset("json", data_files=PREFERENCE_DATA_FILE)
        preference_dataset = dataset["train"]
        logger.info(f"Loaded preference dataset: {preference_dataset}")
        # Let's use the entire dataset for DPO training for maximum signal
        # Splitting might be needed only for monitoring, but DPO loss itself
        # is the primary signal.
        dpo_train_dataset = preference_dataset
        dpo_eval_dataset = None # No separate evaluation split for DPO loss
        logger.info(f"DPO training dataset size: {len(dpo_train_dataset)}")
    except Exception as e:
        logger.error(f"Error loading preference data from {PREFERENCE_DATA_FILE}: {e}")
        logger.info("Please run generate_preference_data.py first to create this file.")
        exit(1)

    try:
        logger.info(f"Loading base model for DPO fine-tuning from {BASE_MODEL_PATH}...")
        # Use AutoModelForSeq2SeqLM for T5 compatibility
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH)
        logger.info("Base model loaded.")

        # TRL's DPOTrainer handles sharing weights where possible if models are same class/config
        logger.info(f"Loading reference model from {REF_MODEL_PATH}...")
        ref_model = AutoModelForSeq2SeqLM.from_pretrained(REF_MODEL_PATH)
        logger.info("Reference model loaded.")

        logger.info(f"Loading tokenizer from {BASE_MODEL_PATH}...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        logger.info("Tokenizer loaded.")

    except Exception as e:
        logger.error(f"Error loading models/tokenizer: {e}")
        exit(1)

    # DPOTrainer requires specific input column names after tokenization
    num_proc = os.cpu_count() // 2 if os.cpu_count() else 1
    logger.info(f"Tokenizing preference data using {num_proc} processes...")

    tokenized_dpo_train_dataset = dpo_train_dataset.map(
        preprocess_for_dpo,
        batched=True,
        remove_columns=list(set(dpo_train_dataset.column_names) - {"chosen", "rejected"}),
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=num_proc,
    )

    tokenized_dpo_eval_dataset = None # No eval dataset for DPO loss monitoring

    logger.info(f"Tokenization complete.")
    logger.info(f"Tokenized DPO training dataset: {tokenized_dpo_train_dataset}")

    gc.collect()
    torch.cuda.empty_cache()


    total_train_samples = len(tokenized_dpo_train_dataset)
    effective_batch_size = TRAIN_BATCH_SIZE_DPO * GRADIENT_ACCUMULATION_STEPS_DPO
    steps_per_epoch = (total_train_samples + effective_batch_size - 1) // effective_batch_size
    total_training_steps = steps_per_epoch * NUM_EPOCHS_DPO
    eval_steps = max(100, steps_per_epoch // 10)
    save_steps = eval_steps * 2 
    logging_steps = max(50, eval_steps // 10)

    logger.info(f"DPO training effective batch size: {effective_batch_size}")
    logger.info(f"DPO training steps per epoch: {steps_per_epoch}")
    logger.info(f"DPO training total steps: {total_training_steps}")
    logger.info(f"DPO logging steps: {logging_steps}")
    logger.info(f"DPO save steps: {save_steps}")

    training_args_dpo = DPOConfig(
        output_dir=OUTPUT_DIR_DPO,
        num_train_epochs=NUM_EPOCHS_DPO,
        per_device_train_batch_size=TRAIN_BATCH_SIZE_DPO,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS_DPO,
        learning_rate=LEARNING_RATE_DPO,
        warmup_ratio=WARMUP_RATIO_DPO,
        weight_decay=WEIGHT_DECAY_DPO,
        beta=DPO_BETA,
        optim="adamw_torch",
        lr_scheduler_type=SCHEDULER_TYPE,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=SAVE_TOTAL_LIMIT_DPO,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=REPORT_TO_DPO,
        seed=SEED,
        data_seed=SEED,
        logging_dir=f"{OUTPUT_DIR_DPO}/logs",remove_unused_columns=False
    )

    # DPOTrainer uses its own DataCollator internally optimized for DPO data structure
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args_dpo,
        train_dataset=tokenized_dpo_train_dataset,
        eval_dataset=tokenized_dpo_eval_dataset,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training...")
    train_result = trainer.train()
    logger.info("DPO training complete.")

    logger.info(f"Saving DPO trained model to {OUTPUT_DIR_DPO}...")
    trainer.save_model(OUTPUT_DIR_DPO)
    logger.info("DPO trained model saved.")

    metrics = train_result.metrics
    trainer.log_metrics("dpo_train", metrics)
    trainer.save_metrics("dpo_train", metrics)
    trainer.save_state()
    logger.info(f"DPO Training Metrics: {metrics}")

    del model, ref_model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()