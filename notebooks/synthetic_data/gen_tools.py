import re
import nltk
from bs4 import BeautifulSoup
from datasets import Dataset
import google.generativeai as genai
from tqdm.auto import tqdm
import os
from dotenv import load_dotenv
import json
import time
from nltk.tokenize import word_tokenize
import google.api_core.exceptions
import numpy as np

TASK_PROMPT = "<LM> Сократи текст: "
FRED_T5_MODEL_NAME = "ai-forever/FRED-T5-large"
INPUT_TOKEN_LIMIT_FRED_T5 = 1024

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME_PRIMARY = "gemini-2.0-flash"
MIN_TEXT_TOKENS_FOR_SUMMARIZATION = 100

GEMINI_GENERATION_CONFIG = genai.GenerationConfig(
    candidate_count=3,
    max_output_tokens=300,
    temperature=0.8
)
MIN_DELAY_BETWEEN_REQUESTS = 4
QUANTILE_LEVELS = [0.25, 0.50, 0.75]
SAVE_EVERY_N_ITEMS = 10
RETRY_DELAY_SECONDS = 10
MAX_RETRIES = 3

SUMMARIZATION_PROMPT_TEMPLATE = """
Ты — эксперт по созданию кратких изложений текстов (саммари). Твоя задача — максимально сократить текст, выделив только основную суть.
Саммари должно быть на русском языке, грамматически корректным, не искажать факты и при этом не быть копией исходного текста.
Результат должен быть в три раза короче исходного текста, не более 40% от длины исходного текста.

Текст:
{text_to_summarize}

Саммари:"""


def clean_text(raw_text):
    if not isinstance(raw_text, str):
        return ""

    text = raw_text

    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)

    # Remove CSS @font-face definitions
    text = re.sub(r'@font-face\s*\{[^\}]*\}', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove common MS Word style blocks
    text = re.sub(r'\.(MsoNormal|MsoChpDefault|MsoPapDefault|apple-converted-space)\s*\{[^\}]*\}', '', text,
                  flags=re.DOTALL | re.IGNORECASE)

    # Remove CSS page section definitions
    text = re.sub(r'@page\s+Section\d+\s*\{[^\}]*\}', '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove general C-style comments /* ... */
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.DOTALL)

    # Remove inline JavaScript and CSS inside <script> or <style> tags
    soup = BeautifulSoup(text, "html.parser")
    for s_tag in soup(['script', 'style']):
        s_tag.decompose()

    # Extract plain text from HTML
    text = soup.get_text(separator=' ', strip=True)

    # Remove URLs (http, https, www)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs

    # Remove emails
    text = re.sub(r'\b[\w.-]+?@\w+?\.\w+?\b', '', text)  # Remove email addresses

    # Remove leftover HTML entities (e.g., &nbsp;)
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # Remove HTML entities

    # Remove dates (basic ISO and common formats)
    text = re.sub(r'\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b', '', text)  # Remove dates like 12.05.2023 or 12/05/23

    # Remove time patterns (e.g., 13:45, 9:00 AM)
    text = re.sub(r'\b\d{1,2}:\d{2}(?:\s?[APMapm]{2})?\b', '', text)  # Remove times like 14:00, 9:30 AM

    # Remove phone numbers
    text = re.sub(r'\+?\d[\d\s().-]{6,}\d', '', text)  # Remove phone numbers

    # Remove currency amounts (e.g., $20, 500₽, €300)
    text = re.sub(r'[$€₽¥£]\s?\d+[.,]?\d*', '', text)  # Remove money values
    text = re.sub(r'\d+[.,]?\d*\s?(USD|EUR|RUB|GBP|JPY)', '', text, flags=re.IGNORECASE)  # Remove money with currency

    # Remove multiple whitespace characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace

    return text


def smart_truncate_text(text, tokenizer_instance, max_tokens_for_text_part):
    token_ids = tokenizer_instance(text, truncation=False, add_special_tokens=False)['input_ids']
    if len(token_ids) <= max_tokens_for_text_part:
        return text

    sentences = nltk.sent_tokenize(text, language='russian')
    truncated_text_parts = []
    current_token_count = 0
    for sentence in sentences:
        sentence_token_ids = tokenizer_instance(sentence, truncation=False, add_special_tokens=False)['input_ids']
        if current_token_count + len(sentence_token_ids) > max_tokens_for_text_part:
            break
        truncated_text_parts.append(sentence)
        current_token_count += len(sentence_token_ids)

    result = " ".join(truncated_text_parts).strip()
    return result

def preprocess_and_filter_dataset_with_exact_deduplication(
        dataset_split,
        tokenizer_instance,
        max_tokens_for_processing,
        min_tokens_for_output
):
    print(f"Starting preprocessing. Initial size: {len(dataset_split)}")
    print(f"Targeting max text tokens for processing: {max_tokens_for_processing}")

    processed_data = []
    unique_processed_texts = set()
    exact_duplicates_found = 0

    for example in tqdm(dataset_split, desc="Cleaning, Truncating, and Exact Deduplicating"):
        original_text = example.get("text", "")
        file_identifier = example.get("file", "unknown_file")

        cleaned_text = clean_text(original_text)
        truncated_text_for_processing = smart_truncate_text(
            cleaned_text,
            tokenizer_instance,
            max_tokens_for_processing
        )

        # Проверка на точный дубликат ПЕРЕД подсчетом токенов и другими проверками
        if truncated_text_for_processing in unique_processed_texts:
            exact_duplicates_found += 1
            continue  # Пропускаем этот текст, так как он уже был добавлен

        # Если текст не дубликат, добавляем его в set
        unique_processed_texts.add(truncated_text_for_processing)

        final_text_token_ids = \
        tokenizer_instance(truncated_text_for_processing, truncation=False, add_special_tokens=False)['input_ids']
        final_text_token_count = len(final_text_token_ids)

        if final_text_token_count >= min_tokens_for_output:
            processed_data.append({
                "file": file_identifier,
                "processed_text": truncated_text_for_processing,
                "processed_text_tokens": final_text_token_count
            })

    filtered_hf_dataset = Dataset.from_list(processed_data)
    print(f"Finished preprocessing. Exact duplicates found and skipped: {exact_duplicates_found}")
    print(f"Filtered dataset size after exact deduplication: {len(filtered_hf_dataset)}")
    return filtered_hf_dataset


def get_simple_stats(text, tokenizer_instance):
    if not text or not isinstance(text, str):
        return {"Chars": 0, "Words": 0, "Tokens": 0, "IsNonEmpty": False}
    return {
        "Chars": len(text),
        "Words": len(word_tokenize(text, language="russian")),
        "Tokens": len(tokenizer_instance.encode(text, add_special_tokens=False)),
        "IsNonEmpty": True
    }

def calculate_compression_ratios(source, summary):
    return {
        f"Compression_{k}": source[k] / summary[k]
        for k in ["Chars", "Words", "Tokens"]
    }

def generate_summaries_resumable_full_stats(dataset, model, tokenizer_instance, PROCESSED_INDICES_FILE, OUTPUT_JSONL_FILE):
    if model is None:
        print("Gemini model not initialized. Skipping generation.")
        return

    if 'GEMINI_GENERATION_CONFIG' not in globals() and 'GEMINI_GENERATION_CONFIG' not in locals():
        raise NameError("GEMINI_GENERATION_CONFIG is not defined globally.")

    processed_indices = set()
    if os.path.exists(PROCESSED_INDICES_FILE):
        with open(PROCESSED_INDICES_FILE) as f:
            processed_indices = {int(line.strip()) for line in f if line.strip()}
    print(f"Loaded {len(processed_indices)} already processed indices.")

    items_processed = 0
    total_time = 0
    items_with_no_summaries = []
    num_candidates = GEMINI_GENERATION_CONFIG.candidate_count
    all_candidate_stats = {
        "Chars_abs": [], "Words_abs": [], "Tokens_abs": [],
        "Chars_ratios": [], "Words_ratios": [], "Tokens_ratios": [],
        "NonEmptyCount": 0
    }
    source_stats_list = []
    success_dist = [0] * (num_candidates + 1)
    fatal_error_occurred = False

    indices_to_process = [i for i in range(len(dataset)) if i not in processed_indices]
    if not indices_to_process:
        print("No new items to process.")
        return

    print(f"Attempting to process {len(indices_to_process)} items in this run.")

    last_request_time = None

    try:
        with open(OUTPUT_JSONL_FILE, "a", encoding="utf-8") as fout, open(PROCESSED_INDICES_FILE, "a") as findices:
            for idx in tqdm(indices_to_process, desc="Generating Summaries & Full Stats"):
                if fatal_error_occurred:
                    break

                record = dataset[idx]
                if not isinstance(record, dict) or "processed_text" not in record:
                    print(f"Skipping invalid record at index {idx}: {record}")
                    continue

                text = record["processed_text"]
                source_stats = get_simple_stats(text, tokenizer_instance)
                if source_stats["IsNonEmpty"]:
                    source_stats_list.append(source_stats)

                prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(text_to_summarize=text)
                summaries = [""] * num_candidates
                success_count = 0

                for attempt in range(MAX_RETRIES + 1):
                    if last_request_time is not None:
                        elapsed = time.time() - last_request_time
                        if elapsed < MIN_DELAY_BETWEEN_REQUESTS:
                            time.sleep(MIN_DELAY_BETWEEN_REQUESTS - elapsed)

                    try:
                        start_time = time.time()
                        response = model.generate_content(
                            prompt,
                            generation_config=GEMINI_GENERATION_CONFIG,
                            request_options={'timeout': 120}
                        )
                        last_request_time = time.time()
                        total_time += (last_request_time - start_time)
                        for i, candidate in enumerate(response.candidates or []):
                            if i < num_candidates and candidate.content and candidate.content.parts:
                                summaries[i] = candidate.content.parts[0].text
                                if summaries[i]:
                                    success_count += 1
                        break  # Успешно — выходим из retry-цикла
                    except (google.api_core.exceptions.PermissionDenied,
                            google.api_core.exceptions.ResourceExhausted,
                            google.api_core.exceptions.InvalidArgument) as fatal:
                        print(f"FATAL API Error for index {idx}: {fatal}. Stopping.")
                        fatal_error_occurred = True
                        break
                    except Exception as e:
                        print(f"Error for index {idx} (attempt {attempt+1}/{MAX_RETRIES+1}): {e}")
                        if attempt < MAX_RETRIES:
                            time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                        else:
                            last_request_time = time.time()

                success_dist[success_count] += 1
                if success_count == 0:
                    items_with_no_summaries.append(idx)
                    continue

                for i in range(num_candidates):
                    stats = get_simple_stats(summaries[i], tokenizer_instance)
                    if stats["IsNonEmpty"] and source_stats["IsNonEmpty"]:
                        all_candidate_stats["NonEmptyCount"] += 1
                        for key in ["Chars", "Words", "Tokens"]:
                            all_candidate_stats[f"{key}_abs"].append(stats[key])
                        ratios = calculate_compression_ratios(source_stats, stats)
                        for key in ["Chars", "Words", "Tokens"]:
                            all_candidate_stats[f"{key}_ratios"].append(ratios[f"Compression_{key}"])

                fout.write(json.dumps({
                    "original_dataset_index": idx,
                    "file": record.get("file", "N/A"),
                    "processed_text": text,
                    "summaries": summaries
                }, ensure_ascii=False) + "\n")
                findices.write(f"{idx}\n")
                processed_indices.add(idx)
                items_processed += 1

                if items_processed % SAVE_EVERY_N_ITEMS == 0:
                    fout.flush(); findices.flush()
    finally:
        print(f"\n--- Generation Finished for this Run ---")
        print(f"Processed: {items_processed}, Total Time: {total_time:.2f}s, Avg/Item: {total_time/items_processed if items_processed else 0:.2f}s")

        if source_stats_list:
            print("\nSource Stats:")
            for metric in ["Chars", "Words", "Tokens"]:
                values = [s[metric] for s in source_stats_list if s["IsNonEmpty"] and s[metric] > 0]
                if values:
                    print(f"  {metric}: Avg={np.mean(values):.2f}, Min={np.min(values)}, Max={np.max(values)}")

        print(f"\nAll Summary Candidates ({all_candidate_stats['NonEmptyCount']} non-empty):")
        if all_candidate_stats["NonEmptyCount"] > 0:
            for metric in ["Chars", "Words", "Tokens"]:
                abs_vals = np.array(all_candidate_stats[f"{metric}_abs"])
                ratio_vals = np.array(all_candidate_stats[f"{metric}_ratios"])

                print(f"  {metric}:")
                print(f"    Avg={np.mean(abs_vals):.2f}, Min={np.min(abs_vals)}, Max={np.max(abs_vals)}")
                print(f"    Comp.: Avg={np.nanmean(ratio_vals):.3f}, Min={np.nanmin(ratio_vals):.3f}, Max={np.nanmax(ratio_vals):.3f}")
                for q in QUANTILE_LEVELS:
                    print(f"    {q*100:.0f}% Quantile: Abs={np.quantile(abs_vals, q):.2f}, Comp.={np.nanquantile(ratio_vals, q):.3f}")
        else:
            print("  No summary data.")

        print(f"\nItems with no summaries: {items_with_no_summaries}")
        print("Success distribution:")
        for c, n in enumerate(success_dist):
            if n:
                print(f"  {c} summaries: {n} items")


def create_final_huggingface_dataset_from_jsonl(jsonl_file_path, output_hf_dataset_path):
    if not os.path.exists(jsonl_file_path):
        print(f"JSONL file not found: {jsonl_file_path}. Cannot create dataset.")
        return None

    all_data_from_jsonl = []
    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                all_data_from_jsonl.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Skipping malformed line in {jsonl_file_path}: {line.strip()}")

    if not all_data_from_jsonl:
        print("No data loaded from JSONL file. Dataset will be empty.")
        return None

    dataset_dict_for_hf = {
        "file": [d.get("file", "") for d in all_data_from_jsonl],
        "text": [d.get("processed_text", "") for d in all_data_from_jsonl],
        "summaries": [d.get("summaries", ["", "", ""]) for d in all_data_from_jsonl]
    }

    final_hf_dataset = Dataset.from_dict(dataset_dict_for_hf)
    print(f"Created Hugging Face dataset with {len(final_hf_dataset)} examples from JSONL.")

    if output_hf_dataset_path:
        print(f"Saving Hugging Face dataset to {output_hf_dataset_path}...")
        final_hf_dataset.save_to_disk(output_hf_dataset_path)
        print("Hugging Face dataset saved.")

    return final_hf_dataset
