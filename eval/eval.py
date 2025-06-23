import logging
import os
import json
from time import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    GenerationConfig
)
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
from transformers.trainer_utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, choices=['nat', 'synth'], required=True, help="Evaluation mode: 'natural' or 'synth'")
parser.add_argument('--model_path', type=str, required=True, help="Path to the model")
parser.add_argument('--eval_name', type=str, required=True, help="Evaluation name (used for file naming)")
args = parser.parse_args()

mode = args.mode
MODEL_PATH = args.model_path
EVAL_NAME = args.eval_name

if mode == "nat":
    TEST_DATA_PATH = "data/filtered_test"
    OUTPUT_DIR = "./eval/res"
else:
    TEST_DATA_PATH = "data/synth_test"
    OUTPUT_DIR = "./eval/res_synth"

OUTPUT_FILE = OUTPUT_DIR + f"/metrics_{EVAL_NAME}.json"
PREDICTIONS_FILE = OUTPUT_DIR + f"/output_{EVAL_NAME}.json"
TOKENIZER_PATH = MODEL_PATH

EVAL_BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABSE_MODEL_NAME = 'sentence-transformers/LaBSE'

GENERATION_CONFIG = GenerationConfig(
    max_length=MAX_TARGET_LENGTH + 1,
    num_beams=NUM_BEAMS,
    min_length=MIN_GENERATION_LENGTH,
    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
    do_sample=False,
    early_stopping=True
)

def preprocess_for_generation(examples, tokenizer):
    tokenized = tokenizer(
        [TASK_PROMPT + text for text in examples["text"]],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False, # Defer padding to DataCollator
    )
    return tokenized

def generate_predictions(model, tokenizer, dataset, batch_size):
    tokenized_dataset = dataset.map(
        preprocess_for_generation,
        batched=True,
        batch_size=batch_size * 2,
        fn_kwargs={"tokenizer": tokenizer},
    )
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=data_collator)

    all_predictions = []
    logger.info(f"Generating predictions (batch_size: {batch_size}, num_beams: {NUM_BEAMS})...")
    generation_start_time = time()
    for batch in tqdm(dataloader, desc="Generating"):
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                generation_config=GENERATION_CONFIG
            )
        preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_predictions.extend(preds)
        del input_ids, attention_mask, generated_ids
        torch.cuda.empty_cache()
    generation_time = time() - generation_start_time
    logger.info(f"Generation time: {generation_time:.2f} seconds.")
    return all_predictions

def evaluate_predictions(predictions, references, inputs, tokenizer, output_file, predictions_file):
    logger.info("Metrics calculation...")
    metrics_start_time = time()
    results = {}

    cleaned_predictions = [p.strip() for p in predictions]
    cleaned_references = [r.strip() for r in references]
    # logger.info("Filtering for compression (summary shorter than input)...")
    # initial_count = len(cleaned_predictions)

    # filtered_data = [
    #     (pred, ref, inp)
    #     for pred, ref, inp in zip(cleaned_predictions, cleaned_references, inputs)
    #     if len(tokenizer(pred)["input_ids"]) < len(tokenizer(inp)["input_ids"])
    # ]
    filtered_inputs = inputs
    # if filtered_data:
    #     cleaned_predictions, cleaned_references, filtered_inputs = zip(*filtered_data)
    # else:
    #     cleaned_predictions, cleaned_references, filtered_inputs = [], [], []

    # filtered_count = len(cleaned_predictions)
    # logger.info(f"Remaining after compression filter: {filtered_count} out of {initial_count} ({(filtered_count / initial_count * 100):.2f}%)")

    try:
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        meteor = evaluate.load("meteor")
        bertscore = evaluate.load("bertscore")
        chrf = evaluate.load("chrf")
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        exit(1)

    try:
        rouge_result = rouge.compute(predictions=cleaned_predictions, references=cleaned_references, use_stemmer=True)
        results.update({
            'rouge1': rouge_result['rouge1'],
            'rouge2': rouge_result['rouge2'],
            'rougeL': rouge_result['rougeLsum']
        })
    except Exception as e:
        logger.error(f"Error evaluating ROUGE: {e}")
        results.update({'rouge1': 0, 'rouge2': 0, 'rougeL': 0})

    try:
        bleu_result = bleu.compute(predictions=cleaned_predictions, references=[[r] for r in cleaned_references])
        results['bleu'] = bleu_result['bleu']
    except Exception as e:
        logger.error(f"Error evaluating BLEU: {e}")
        results['bleu'] = 0

    try:
        meteor_result = meteor.compute(predictions=cleaned_predictions, references=cleaned_references)
        results['meteor'] = meteor_result['meteor']
    except Exception as e:
        logger.error(f"Error evaluating METEOR: {e}")
        results['meteor'] = 0

    try:
        chrf_result = chrf.compute(predictions=cleaned_predictions, references=[[r] for r in cleaned_references], word_order=2)
        results['chrf'] = chrf_result['score']
    except Exception as e:
        logger.error(f"Error evaluating CHRF: {e}")
        results['chrf'] = 0

    try:
        bert_result = bertscore.compute(predictions=cleaned_predictions, references=cleaned_references, lang="ru", device=DEVICE)
        results['bertscore_f1'] = np.mean(bert_result['f1'])
    except Exception as e:
        logger.error(f"Error evaluating BERTScore: {e}")
        results['bertscore_f1'] = 0

    try:
        labse_model = SentenceTransformer(LABSE_MODEL_NAME, device=DEVICE)
        with torch.no_grad():
            pred_embs = labse_model.encode(cleaned_predictions, convert_to_tensor=True, device=DEVICE, show_progress_bar=True)
            ref_embs = labse_model.encode(cleaned_references, convert_to_tensor=True, device=DEVICE, show_progress_bar=True)
        cosine_sim = util.pytorch_cos_sim(pred_embs, ref_embs)
        results['labse_similarity'] = torch.diag(cosine_sim).mean().item()
    except Exception as e:
        logger.error(f"Error evaluating LaBSE: {e}")
        results['labse_similarity'] = 0

    pred_tokenized = tokenizer(cleaned_predictions, truncation=False, padding=False)["input_ids"]
    input_token_lens = [len(tokenizer(inp)["input_ids"]) for inp in filtered_inputs]
    pred_token_lens = [len(tokens) for tokens in pred_tokenized]

    results['gen_len_mean'] = np.mean(pred_token_lens)
    results['gen_len_std'] = np.std(pred_token_lens)
    compression_ratios = [i / p for i, p in zip(input_token_lens, pred_token_lens)]
    results['compression_ratio_mean'] = np.mean(compression_ratios)
    results['compression_ratio_std'] = np.std(compression_ratios)

    metrics_time = time() - metrics_start_time
    logger.info(f"Metrics time: {metrics_time:.2f} seconds.")
    results = {k: round(v, 3) for k, v in results.items()}

    logger.info(f"Final results:\n{json.dumps(results, indent=4)}")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info("Saving predictions...")
    predictions_data = [
        {
            "id": i,
            "input_text": inputs[i],
            "reference_summary": references[i],
            "predicted_summary": predictions[i]
        }
        for i in range(len(predictions))
    ]
    os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=4)

    logger.info("Evaluation completed.")


if __name__ == "__main__":
    set_seed(SEED)
    utils.set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE).eval()

    test_dataset = load_from_disk(TEST_DATA_PATH)
    input_texts = test_dataset["text"]
    reference_summaries = test_dataset["summary"]

    predictions = generate_predictions(model, tokenizer, test_dataset, EVAL_BATCH_SIZE)
    gc.collect()
    torch.cuda.empty_cache()

    evaluate_predictions(
        predictions=predictions,
        references=reference_summaries,
        inputs=input_texts,
        tokenizer=tokenizer,
        output_file=OUTPUT_FILE,
        predictions_file=PREDICTIONS_FILE
    )


# import json
# records = []
# with open("eval/baseline_2.jsonl", "r", encoding="utf-8") as f:
#     for line in f:
#         records.append(json.loads(line))
# input_texts = [rec["input_text"].strip() for rec in records]
# reference_summaries = [rec["reference_summary"].strip() for rec in records]
# predictions = [rec["predicted_summary"].strip() for rec in records]
#
# with open("eval/res/output_synth_DPO.json", "r", encoding="utf-8") as f:
#     records = json.load(f)
# input_texts = [rec["input_text"] for rec in records]
# reference_summaries = [rec["reference_summary"] for rec in records]
# predictions = [rec["predicted_summary"] for rec in records]