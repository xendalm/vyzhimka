import os
import logging
import gc
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from trl import DPOTrainer, DPOConfig
from common import utils
from common.const import (
    TASK_PROMPT,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    SEED,
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

DPO_BETA = 0.1
NUM_EPOCHS_DPO = 3
TRAIN_BATCH_SIZE_DPO = 8
GRADIENT_ACCUMULATION_STEPS_DPO = 2
LEARNING_RATE_DPO = 1e-6
WEIGHT_DECAY_DPO = 0.0
WARMUP_RATIO_DPO = 0.05
SAVE_TOTAL_LIMIT_DPO = 2
REPORT_TO_DPO = "tensorboard"
SCHEDULER_TYPE = "cosine"

if __name__ == "__main__":
    utils.set_seed(SEED)

    try:
        dataset = load_dataset("json", data_files=PREFERENCE_DATA_FILE)
        preference_dataset = dataset["train"]

        logger.info(f"Loaded dataset with {len(preference_dataset)} examples.")

        preference_dataset = preference_dataset.map(
            lambda example: {
                "prompt": TASK_PROMPT + example["text"],
                "chosen": example["chosen"],
                "rejected": example["rejected"],
            },
            remove_columns=preference_dataset.column_names,
        )

        logger.info(f"Prepared dataset for DPO with fields: {preference_dataset.features}")
    except Exception as e:
        logger.error(f"Error loading preference data from {PREFERENCE_DATA_FILE}: {e}")
        logger.info("Please run generate_preference_data.py first to create this file.")
        exit(1)

    try:
        logger.info("Loading base and reference models...")
        model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL_PATH)
        ref_model = AutoModelForSeq2SeqLM.from_pretrained(REF_MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        logger.info("Models and tokenizer loaded.")
    except Exception as e:
        logger.error(f"Error loading models/tokenizer: {e}")
        exit(1)

    gc.collect()
    torch.cuda.empty_cache()

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
        save_steps=200,
        logging_steps=20,
        save_total_limit=SAVE_TOTAL_LIMIT_DPO,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to=REPORT_TO_DPO,
        seed=SEED,
        data_seed=SEED,
        logging_dir=f"{OUTPUT_DIR_DPO}/logs",
        # max_prompt_length=MAX_INPUT_LENGTH,
        # max_completion_length=MAX_TARGET_LENGTH,
        remove_unused_columns=False,  # критично для DPOTrainer
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args_dpo,
        train_dataset=preference_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
    )

    logger.info("Starting DPO training...")
    train_result = trainer.train()
    logger.info("DPO training complete.")

    logger.info(f"Saving model to {OUTPUT_DIR_DPO}...")
    trainer.save_model(OUTPUT_DIR_DPO)

    metrics = train_result.metrics
    trainer.log_metrics("dpo_train", metrics)
    trainer.save_metrics("dpo_train", metrics)
    trainer.save_state()
    logger.info(f"Training metrics: {metrics}")

    del model, ref_model, tokenizer, trainer
    gc.collect()
    torch.cuda.empty_cache()