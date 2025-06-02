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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

MODEL_PATH = "fred-t5_summarization_dpo"
TOKENIZER_PATH = MODEL_PATH
TEST_DATA_PATH = "./data/filtered_test"
OUTPUT_FILE = "./eval/results/evaluation_metrics_dpo.json"
PREDICTIONS_FILE = "./eval/results/predictions_output_dpo.json"

EVAL_BATCH_SIZE = 55
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABSE_MODEL_NAME = 'sentence-transformers/LaBSE'

GENERATION_CONFIG = GenerationConfig(
    # length_penalty=-2,
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

if __name__ == "__main__":
    utils.set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(DEVICE).eval()
    labse_model = SentenceTransformer(LABSE_MODEL_NAME, device=DEVICE)

    test_dataset = load_from_disk(TEST_DATA_PATH)

    test_dataset_tokenized = test_dataset.map(
        preprocess_for_generation,
        batched=True,
        batch_size=EVAL_BATCH_SIZE * 2,
        fn_kwargs={"tokenizer": tokenizer},
    )
    test_dataset_tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    eval_dataloader = DataLoader(test_dataset_tokenized, batch_size=EVAL_BATCH_SIZE, collate_fn=data_collator)

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

    gc.collect()
    torch.cuda.empty_cache()

    logger.info("Metrics calculation...")
    metrics_start_time = time()
    results = {}

    cleaned_predictions = [p.strip() for p in all_predictions]
    cleaned_references = [r.strip() for r in all_references]

    try:
        rouge = evaluate.load("rouge")
        bleu = evaluate.load("bleu")
        meteor = evaluate.load("meteor")
        bertscore = evaluate.load("bertscore")
        chrf = evaluate.load("chrf")
    except Exception as e:
        logger.error(f"Error loading metrics: {e}")
        exit(1)

    logger.info("Evaluating ROUGE...")
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

    logger.info("Evaluating BLEU...")
    try:
        bleu_result = bleu.compute(predictions=cleaned_predictions, references=[[ref] for ref in cleaned_references])
        results['bleu'] = bleu_result['bleu']
    except Exception as e:
        logger.error(f"Error evaluating BLEU: {e}")
        results['bleu'] = 0

    logger.info("Evaluating METEOR...")
    try:
        meteor_result = meteor.compute(predictions=cleaned_predictions, references=cleaned_references)
        results['meteor'] = meteor_result['meteor']
    except Exception as e:
        logger.error(f"Error evaluating METEOR: {e}")
        results['meteor'] = 0

    logger.info("Evaluating CHRF++...")
    try:
        chrf_result = chrf.compute(predictions=cleaned_predictions, references=[[ref] for ref in cleaned_references], word_order=2)
        results['chrf'] = chrf_result['score']
    except Exception as e:
        logger.error(f"Error evaluating CHRF: {e}")
        results['chrf'] = 0

    logger.info("Evaluating BERTScore...")
    try:
        bert_result = bertscore.compute(predictions=cleaned_predictions, references=cleaned_references, lang="ru", device=DEVICE)
        results['bertscore_f1'] = np.mean(bert_result['f1'])
    except Exception as e:
        logger.error(f"Error evaluating BERTScore: {e}")
        results['bertscore_f1'] = 0

    logger.info("Evaluating LaBSE similarity...")
    try:
        with torch.no_grad():
            pred_embs = labse_model.encode(cleaned_predictions, convert_to_tensor=True, device=DEVICE, show_progress_bar=True, batch_size=EVAL_BATCH_SIZE * 4)
            ref_embs = labse_model.encode(cleaned_references, convert_to_tensor=True, device=DEVICE, show_progress_bar=True, batch_size=EVAL_BATCH_SIZE * 4)
        cosine_sim = util.pytorch_cos_sim(pred_embs, ref_embs)
        results['labse_similarity'] = torch.diag(cosine_sim).mean().item()
    except Exception as e:
        logger.error(f"Error evaluating LaBSE: {e}")
        results['labse_similarity'] = 0

    logger.info("Calculating average generation length...")
    pred_tokens = tokenizer(cleaned_predictions, truncation=False, padding=False)["input_ids"]
    results['gen_len_mean'] = np.mean([len(t) for t in pred_tokens])
    results['gen_len_std'] = np.std([len(t) for t in pred_tokens])


    logger.info("Calculating average compression ratio...")
    results['compression_ratio_mean'] = np.mean([len(t) / len(r) for t, r in zip(cleaned_predictions, cleaned_references)])
    results['compression_ratio_std'] = np.std([len(t) / len(r) for t, r in zip(cleaned_predictions, cleaned_references)])

    metrics_time = time() - metrics_start_time
    logger.info(f"Metrics time: {metrics_time:.2f} seconds.")

    results = {k: round(v, 3) for k, v in results.items()}
    logger.info(f"Final results:\n{json.dumps(results, indent=4)}")

    logger.info("Saving results...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    logger.info("Saving predictions...")
    predictions_data = [
        {
            "id": i,
            "input_text": all_input_texts[i],
            "reference_summary": all_references[i],
            "predicted_summary": all_predictions[i]
        }
        for i in range(len(all_predictions))
    ]
    os.makedirs(os.path.dirname(PREDICTIONS_FILE), exist_ok=True)
    with open(PREDICTIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=4)

    logger.info("Done.")

# {                                                                                                    
#     "rouge1": 0.296,                                                                                 
#     "rouge2": 0.17,                                                                                  
#     "rougeL": 0.287,                                                                                 
#     "bleu": 0.143,                                                                                   
#     "meteor": 0.424,                                                                                 
#     "chrf": 51.962,                                                                                  
#     "bertscore_f1": 0.772,                                                                           
#     "labse_similarity": 0.869,                                                                       
#     "gen_len": 226.194                                                                               
# }  