import os
import logging
import gc
import json
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    GenerationConfig
)
from tqdm.auto import tqdm
import nltk

from common import utils
from common.const import (
    TASK_PROMPT,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    MIN_GENERATION_LENGTH,
    NO_REPEAT_NGRAM_SIZE,
    SEED
)
from datasets import load_from_disk

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

SUMMARIZATION_MODEL_PATH = "fred-t5_summarization_combined"
SCORING_MODEL_NAME = "google/seahorse-large-q6"
TRAIN_DATA_PATH = "data/combined_train"
OUTPUT_PREFERENCE_DATA_FILE = "data/preference_data_conciseness_2.jsonl"

NUM_GENERATIONS_PER_TEXT = 6
GENERATION_BATCH_SIZE = 45
SUM_GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_TARGET_LENGTH + 1,
    num_beams=NUM_GENERATIONS_PER_TEXT,
    min_length=MIN_GENERATION_LENGTH,
    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    early_stopping=True,
    do_sample=False,
    num_return_sequences=NUM_GENERATIONS_PER_TEXT,
)

SCORING_BATCH_SIZE = 120
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEAHORSE_FORMAT_GOOGLE = "premise: {} hypothesis: {}"
SEAHORSE_ZERO_TOKEN = '▁0'
SEAHORSE_ONE_TOKEN = '▁1'

def truncate_incomplete_sentence(summary_text):
    if not summary_text or not summary_text.strip():
        return ""

    sentences = nltk.sent_tokenize(summary_text, language='russian')
    last_sentence = sentences[-1].strip()
    sentence_end_punct = ('.', '!', '?')
    if last_sentence.rstrip().endswith(sentence_end_punct):
        truncated_sentences = sentences
    else:
        truncated_sentences = sentences[:-1]

    result = " ".join(truncated_sentences)
    return result.strip()

def preprocess_for_summarization(examples, tokenizer):
    tokenized = tokenizer(
        [TASK_PROMPT + doc for doc in examples["text"]],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False, # Defer padding to DataCollator
    )
    return tokenized

def preprocess_for_scoring(texts, summaries, tokenizer):
    inputs = [SEAHORSE_FORMAT_GOOGLE.format(text, summary) for text, summary in zip(texts, summaries)]
    tokenized = tokenizer(
        inputs,
        max_length=MAX_INPUT_LENGTH + MAX_TARGET_LENGTH + 10,
        truncation=True,
        padding=False, # Defer padding to DataCollator
    )
    return tokenized

if __name__ == "__main__":
    utils.set_seed(SEED)

    try:
        dataset_dict = load_from_disk(TRAIN_DATA_PATH)
        train_dataset = dataset_dict
        logger.info(f"Loaded training dataset: {train_dataset}")
    except Exception as e:
        logger.error(f"Error loading dataset from {TRAIN_DATA_PATH}: {e}")
        exit(1)

    subset_size = 15000
    if len(train_dataset) > subset_size:
        logger.info(f"Using subset of training data ({subset_size} examples) for preference generation...")
        # train_dataset = train_dataset.shuffle(seed=SEED).select(range(subset_size))
        train_dataset = train_dataset.select(range(subset_size))
        logger.info(f"Using subset of training data: {len(train_dataset)} examples")

    try:
        sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_PATH)
        sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_PATH)
        sum_model.to(DEVICE)
        sum_model.eval()
        logger.info("Summarization model loaded.")
    except Exception as e:
        logger.error(f"Error loading summarization model: {e}")
        exit(1)

    sum_dataset_tokenized = train_dataset.map(
        preprocess_for_summarization,
        batched=True,
        fn_kwargs={"tokenizer": sum_tokenizer},
        remove_columns=train_dataset.column_names,
        num_proc=os.cpu_count()//2 if os.cpu_count() else 1
    )
    sum_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    sum_dataloader = DataLoader(
        sum_dataset_tokenized,
        batch_size=GENERATION_BATCH_SIZE,
        collate_fn=DataCollatorForSeq2Seq(sum_tokenizer, model=sum_model, padding=True),
    )

    all_generated_summaries = []
    original_texts = train_dataset["text"]

    logger.info(f"Generating {NUM_GENERATIONS_PER_TEXT} summaries per text (batch_size: {GENERATION_BATCH_SIZE})...")
    for batch in tqdm(sum_dataloader, desc="Generating Summaries"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        with torch.no_grad():
            generated_ids = sum_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=SUM_GENERATION_CONFIG,
                return_dict_in_generate=True,
            )

        decoded_preds_raw = sum_tokenizer.batch_decode(generated_ids.sequences, skip_special_tokens=True)

        truncated_preds = [truncate_incomplete_sentence(pred) for pred in decoded_preds_raw]

        batch_size = input_ids.shape[0]
        generated_summaries_per_text_batch = [
            truncated_preds[j * NUM_GENERATIONS_PER_TEXT : (j + 1) * NUM_GENERATIONS_PER_TEXT]
            for j in range(batch_size)
        ]
        all_generated_summaries.extend(generated_summaries_per_text_batch)

        del input_ids, attention_mask, generated_ids, decoded_preds_raw, truncated_preds
        gc.collect()
        torch.cuda.empty_cache()

    logger.info(f"Finished generating summaries for {len(all_generated_summaries)} texts.")

    del sum_model
    del sum_tokenizer
    del sum_dataloader
    del sum_dataset_tokenized
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Summarization model unloaded. GPU memory freed.")

    try:
        score_tokenizer = AutoTokenizer.from_pretrained(SCORING_MODEL_NAME)
        score_model = AutoModelForSeq2SeqLM.from_pretrained(SCORING_MODEL_NAME)
        score_model.to(DEVICE)
        score_model.eval()
        logger.info("Scoring model loaded.")
        seahorse_zero_token_id = score_tokenizer(SEAHORSE_ZERO_TOKEN, add_special_tokens=False).input_ids[0]
        seahorse_one_token_id = score_tokenizer(SEAHORSE_ONE_TOKEN, add_special_tokens=False).input_ids[0]
        logger.info(f"Seahorse token IDs: 0={seahorse_zero_token_id}, 1={seahorse_one_token_id}")
    except Exception as e:
        logger.error(f"Error loading scoring model: {e}")
        exit(1)

    scoring_pairs_list = []
    for i, summaries in enumerate(all_generated_summaries):
        text = original_texts[i]
        for summary in summaries:
            scoring_pairs_list.append({"text": text, "summary": summary})

    scoring_dataset = Dataset.from_list(scoring_pairs_list)
    scoring_dataset_tokenized = scoring_dataset.map(
        lambda examples: preprocess_for_scoring(examples["text"], examples["summary"], score_tokenizer),
        batched=True,
        remove_columns=scoring_dataset.column_names,
        num_proc=os.cpu_count()//2 if os.cpu_count() else 1
    )
    scoring_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    scoring_dataloader = DataLoader(
        scoring_dataset_tokenized,
        batch_size=SCORING_BATCH_SIZE,
        collate_fn=DataCollatorForSeq2Seq(score_tokenizer, model=score_model, padding=True),
    )

    all_scores_flat = []
    logger.info(f"Scoring generated summaries (batch_size: {SCORING_BATCH_SIZE})...")
    for batch in tqdm(scoring_dataloader, desc="Scoring Summaries"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        with torch.no_grad():
            outputs = score_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            logits = outputs["scores"][0]

            logit_for_0 = logits[:, seahorse_zero_token_id]
            logit_for_1 = logits[:, seahorse_one_token_id]

            scores = torch.sigmoid(logit_for_1 - logit_for_0).cpu().numpy()
            all_scores_flat.extend(scores)

        del input_ids, attention_mask, outputs, logits, logit_for_0, logit_for_1, scores
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Finished scoring summaries.")

    del score_model
    del score_tokenizer
    del scoring_dataloader
    del scoring_dataset
    del scoring_dataset_tokenized
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Scoring model unloaded. GPU memory freed.")

    logger.info(f"Creating preference pairs from scores for {len(original_texts)} original texts...")
    preference_data = []
    score_idx = 0
    for i in tqdm(range(len(original_texts)), desc="Creating Preference Pairs"):
        text = original_texts[i]
        summaries = all_generated_summaries[i]
        # Get scores for this text's summaries from the flattened list
        scores_for_text = all_scores_flat[score_idx : score_idx + NUM_GENERATIONS_PER_TEXT]
        score_idx += NUM_GENERATIONS_PER_TEXT

        summaries_with_scores = list(zip(summaries, scores_for_text))

        summaries_with_scores.sort(key=lambda x: x[1])

        rejected_summary, rejected_score = summaries_with_scores[0]
        chosen_summary, chosen_score = summaries_with_scores[-1]

        scores = [i for _, i in summaries_with_scores]

        if chosen_summary != rejected_summary and (chosen_score - rejected_score) > 0.03:
             preference_data.append({
                 "text": text,
                 "chosen": chosen_summary,
                 "rejected": rejected_summary,
                 "chosen_score": float(chosen_score),
                 "rejected_score": float(rejected_score),
             })
        # else:
        #     if chosen_summary == rejected_summary:
        #         logger.info(f"{i} identical summaries.")
        #     else:
        #         logger.info(f"{i} identical/similar summaries or scores {scores}")


    logger.info(f"Generated {len(preference_data)} preference pairs.")

    logger.info(f"Saving preference data to {OUTPUT_PREFERENCE_DATA_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_PREFERENCE_DATA_FILE), exist_ok=True)
    with open(OUTPUT_PREFERENCE_DATA_FILE, "w", encoding="utf-8") as f:
        for pair in preference_data:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    logger.info("Preference data saved.")

    logger.info("Preference data generation complete.")

