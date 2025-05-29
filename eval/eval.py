import logging
import os
import json
from time import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm
import evaluate 
from sentence_transformers import SentenceTransformer, util
from common.const import (
    TASK_PROMPT,
    MAX_INPUT_LENGTH,
    MAX_TARGET_LENGTH,
    NUM_BEAMS,
    MIN_GENERATION_LENGTH,
    NO_REPEAT_NGRAM_SIZE,
    SEED
)
from common import utils

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# MODEL_PATH = "RussianNLP/FRED-T5-Summarizer"
# MODEL_PATH = "ai-forever/FRED-T5-large"
MODEL_PATH = "/home/student/kkrr/proj/fred-t5_summarization_2"
TOKENIZER_PATH = MODEL_PATH
TEST_DATA_PATH = "./filtered_test"
OUTPUT_FILE = "./evaluation_metrics_natural.json"
PREDICTIONS_FILE = "./predictions_output_natural.json"

EVAL_BATCH_SIZE = 48
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABSE_MODEL_NAME = 'sentence-transformers/LaBSE'

def preprocess_for_generation(examples, tokenizer):
    tokenized = tokenizer(
        [TASK_PROMPT + text for text in examples["text"]],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="longest",
    )
    return tokenized

if __name__ == "__main__":
    utils.set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    labse_model = SentenceTransformer(LABSE_MODEL_NAME, device=DEVICE)

    test_dataset = load_from_disk(TEST_DATA_PATH)

    test_dataset_tokenized = test_dataset.map(
        preprocess_for_generation,
        batched=True,
        batch_size=EVAL_BATCH_SIZE * 2,
        fn_kwargs={"tokenizer": tokenizer},
    )
    test_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    eval_dataloader = DataLoader(test_dataset_tokenized, batch_size=EVAL_BATCH_SIZE)

    all_predictions = []
    all_references = test_dataset['summary']
    all_input_texts = test_dataset['text']

    logger.info(f"Generating predictions (batch_size: {EVAL_BATCH_SIZE}, num_beams: {NUM_BEAMS})...")
    generation_start_time = time()
    for batch in tqdm(eval_dataloader, desc="Generating"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_length=MAX_TARGET_LENGTH + 1, num_beams=NUM_BEAMS,
                min_length=MIN_GENERATION_LENGTH, no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                early_stopping=True,
            )
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_predictions.extend(preds)
        
        del input_ids, attention_mask, generated_ids
        torch.cuda.empty_cache()

    generation_time = time() - generation_start_time
    logger.info(f"Generation time: {generation_time:.2f} seconds.")

    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Metrics calculation...")
    metrics_start_time = time()
    results = {}

    cleaned_predictions = [p.strip() for p in all_predictions]
    cleaned_references = [r.strip() for r in all_references]

    try:
        rouge_metric = evaluate.load("rouge")
        bleu_metric = evaluate.load("bleu")
        meteor_metric = evaluate.load("meteor")
        bertscore_metric = evaluate.load("bertscore")
        chrf_metric = evaluate.load("chrf")
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        exit(1)

    logger.info("Evaluating ROUGE...")
    try:
        rouge_results = rouge_metric.compute(predictions=cleaned_predictions, references=cleaned_references, use_stemmer=True)
        results['rouge1'] = rouge_results['rouge1']
        results['rouge2'] = rouge_results['rouge2']
        results['rougeL'] = rouge_results['rougeLsum']
    except Exception as e:
        logger.error(f"Error evaluating ROUGE: {e}")
        results.update({'rouge1': 0, 'rouge2': 0, 'rougeL': 0})

    logger.info("Evaluating BLEU...")
    try:
        bleu_refs_list = [[ref] for ref in cleaned_references]
        bleu_results = bleu_metric.compute(predictions=cleaned_predictions, references=bleu_refs_list)
        results['bleu'] = bleu_results['bleu']
    except Exception as e:
        logger.error(f"Error evaluating BLEU: {e}")
        results['bleu'] = 0

    logger.info("Evaluating METEOR...")
    try:
        meteor_results = meteor_metric.compute(predictions=cleaned_predictions, references=cleaned_references)
        results['meteor'] = meteor_results['meteor']
    except Exception as e:
        logger.error(f"Error evaluating METEOR: {e}")
        results['meteor'] = 0

    logger.info("Evaluating CHRF...")
    try:
        chrf_refs_list = [[ref] for ref in cleaned_references]
        # Вычисляем chrF++ (word_order=2)
        chrf_results = chrf_metric.compute(predictions=cleaned_predictions, references=chrf_refs_list, word_order=2)
        results['chrf'] = chrf_results['score']
    except Exception as e:
        logger.error(f"Error evaluating METEOR: {e}")
        results['chrf'] = 0

    logger.info(f"Evaluating BERTScore...")
    try:
        bertscore_results = bertscore_metric.compute(
            predictions=cleaned_predictions,
            references=cleaned_references,
            lang="ru",
            device=DEVICE
        )
        results['bertscore_f1'] = np.mean(bertscore_results['f1'])
    except Exception as e:
        logger.error(f"Error evaluating BERTScore: {e}")
        results['bertscore_f1'] = 0


    logger.info(f"Evaluating LaBSE (model: {LABSE_MODEL_NAME})...")
    try:
        with torch.no_grad():
            pred_embeddings = labse_model.encode(cleaned_predictions, convert_to_tensor=True, device=DEVICE, show_progress_bar=True, batch_size=EVAL_BATCH_SIZE*4)
            ref_embeddings = labse_model.encode(cleaned_references, convert_to_tensor=True, device=DEVICE, show_progress_bar=True, batch_size=EVAL_BATCH_SIZE*4)
        cosine_scores = util.pytorch_cos_sim(pred_embeddings, ref_embeddings)
        results['labse_similarity'] = torch.diag(cosine_scores).mean().item()
    except Exception as e:
        logger.error(f"Error evaluating LaBSE: {e}")
        results['labse_similarity'] = 0

    logger.info("Evaluating generation length...")
    pred_tokens = tokenizer(cleaned_predictions, truncation=False, padding=False)["input_ids"]
    results['gen_len'] = np.mean([len(tokens) for tokens in pred_tokens])

    metrics_time = time() - metrics_start_time
    logger.info(f"Metrics time: {metrics_time:.2f} seconds.")

    results = {k: round(v, 3) for k, v in results.items()}
    logger.info(f"Final results: \n{json.dumps(results, indent=4)}")

    logger.info("Saving results...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    predictions_data = []
    for i in range(len(all_predictions)):
        predictions_data.append({
            "id": i,
            "input_text": all_input_texts[i],
            "reference_summary": all_references[i],
            "predicted_summary": all_predictions[i]
        })
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=4)

    logger.info("Done.")
