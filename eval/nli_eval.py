import json
import os
from tqdm import tqdm
import numpy as np
from razdel import sentenize
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import utils
from common.const import (
    SEED
)

utils.set_seed(SEED)

output_paths = [
    "eval/res/output_baseline_2.json",
    "eval/res/output_combined.json",
    "eval/res/output_natural_2.json",
    "eval/res/output_synth.json",
    "eval/res/output_synth_DPO.json",
]

NLI_MODEL_NAME = 'cointegrated/rubert-base-cased-nli-twoway'
OUTPUT_FILE = "eval/factuality/factuality.json"

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(NLI_MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL_NAME).to(device)
model.eval()

label2id = {v.lower(): k for k, v in model.config.id2label.items()}
ENTAILMENT_LABEL_ID = label2id['entailment']
print(f"Model labels: {model.config.id2label}")
print(f"entailment label index: {ENTAILMENT_LABEL_ID}")

def split_into_sentences(text: str) -> list[str]:
    return [s.text for s in sentenize(text) if s.text.strip()]

def calculate_summac_zs_score(source_text: str, summary_text: str) -> float:
    """
    SUMMAC-ZS
    https://arxiv.org/abs/2111.09525
    """
    source_sentences = split_into_sentences(source_text)
    summary_sentences = split_into_sentences(summary_text)

    if not summary_sentences or not source_sentences:
        return 0.0

    max_entailment_scores = []

    for summary_sent in summary_sentences:
        premise_hypothesis_pairs = [(src_sent, summary_sent) for src_sent in source_sentences]
        
        with torch.inference_mode():
            inputs = tokenizer(premise_hypothesis_pairs,
                               padding=True,
                               truncation=True,
                               max_length=512, # Максимальная длина для этой модели
                               return_tensors="pt").to(device)

            logits = model(**inputs).logits
            
            probabilities = torch.softmax(logits, dim=-1)
            entailment_probs = probabilities[:, ENTAILMENT_LABEL_ID].cpu().numpy()

        max_score = np.max(entailment_probs)
        max_entailment_scores.append(max_score)

    final_score = float(np.mean(max_entailment_scores)) if max_entailment_scores else 0.0
    return final_score

factuality_results = {}

for path in tqdm(output_paths, desc="Processing outputs for factuality"):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nli_scores = []
    for entry in tqdm(data, desc=f"Analyzing {os.path.basename(path)}", leave=False):
        input_text = entry['input_text']
        predicted_summary = entry['predicted_summary']

        score = calculate_summac_zs_score(input_text, predicted_summary)
        nli_scores.append(score)

    mean_nli = float(np.mean(nli_scores)) if nli_scores else 0.0
    std_nli = float(np.std(nli_scores)) if nli_scores else 0.0

    factuality_results[os.path.basename(path)] = {
        "mean_summac_zs_score": round(mean_nli, 4),
        "std_summac_zs_score": round(std_nli, 4),
        "num_samples": len(nli_scores)
    }

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(factuality_results, f, ensure_ascii=False, indent=4)