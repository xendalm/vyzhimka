import json
import os
from tqdm import tqdm
import numpy as np
from razdel import tokenize as razdel_tokenize
import re
from collections import deque

output_paths = [
    "eval/res/output_baseline_2.json",
    "eval/res/output_combined.json",
    "eval/res/output_natural_2.json",
    "eval/res/output_synth.json",
    "eval/res/output_synth_DPO.json",
]
OUTPUT_FILE = "eval/plagiarism/plagiarism.json"

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize_words(text: str) -> list[str]:
    return [token.text for token in razdel_tokenize(normalize_text(text))]

def get_ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    
    ngrams = set()
    window = deque(tokens[:n], maxlen=n)
    ngrams.add(tuple(window))
    
    for i in range(n, len(tokens)):
        window.append(tokens[i])
        ngrams.add(tuple(window))
        
    return ngrams

def jaccard_similarity(tokens1: set[str], tokens2: set[str]) -> float:
    union = tokens1 | tokens2
    intersection = tokens1 & tokens2
    return len(intersection) / len(union) if union else 0.0

def calculate_containment_score(source_tokens: set, summary_tokens: set) -> float:
    if not summary_tokens:
        return 0.0
    
    intersection = source_tokens & summary_tokens
    return len(intersection) / len(summary_tokens)

N_GRAM_SIZE = 3 

plagiarism_results = {}

for path in tqdm(output_paths, desc="Processing outputs"):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    jaccard_scores = []
    word_containment_scores = []
    ngram_containment_scores = []

    for entry in data:
        input_text = entry['input_text']
        predicted_summary = entry['predicted_summary']

        source_words = tokenize_words(input_text)
        summary_words = tokenize_words(predicted_summary)

        source_word_set = set(source_words)
        summary_word_set = set(summary_words)
        
        jaccard_scores.append(jaccard_similarity(source_word_set, summary_word_set))

        word_containment_scores.append(calculate_containment_score(source_word_set, summary_word_set))

        source_ngrams = get_ngrams(source_words, N_GRAM_SIZE)
        summary_ngrams = get_ngrams(summary_words, N_GRAM_SIZE)
        ngram_containment_scores.append(calculate_containment_score(source_ngrams, summary_ngrams))

    
    def calculate_stats(scores: list):
        if not scores:
            return 0.0, 0.0
        return float(np.mean(scores)), float(np.std(scores))

    mean_jaccard, std_jaccard = calculate_stats(jaccard_scores)
    mean_word_containment, std_word_containment = calculate_stats(word_containment_scores)
    mean_ngram_containment, std_ngram_containment = calculate_stats(ngram_containment_scores)


    plagiarism_results[os.path.basename(path)] = {
        "jaccard_similarity": {"mean": round(mean_jaccard, 4), "std": round(std_jaccard, 4)},
        "word_containment_score": {"mean": round(mean_word_containment, 4), "std": round(std_word_containment, 4)},
        f"{N_GRAM_SIZE}-gram_containment_score": {"mean": round(mean_ngram_containment, 4), "std": round(std_ngram_containment, 4)},
        "num_samples": len(data)
    }

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(plagiarism_results, f, ensure_ascii=False, indent=4)