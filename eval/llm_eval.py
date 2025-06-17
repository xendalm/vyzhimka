import google.generativeai as genai
import json
from dotenv import load_dotenv
import os
import time
import numpy as np
from tqdm.auto import tqdm
from string import Template
from collections import defaultdict

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

GEMINI_MODEL_NAME = "gemini-2.0-flash"
GENERATION_CONFIG = genai.GenerationConfig(
    temperature=0.2,
    max_output_tokens=2048,
    response_mime_type="application/json"
)
SAFETY_SETTINGS = [
    {"category": cat, "threshold": "BLOCK_NONE"} for cat in [
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT"
    ]
]
DELAY_BETWEEN_REQUESTS = 3
MAX_RETRIES = 3
RETRY_DELAY_S = 10

PROMPT_TEMPLATE = Template("""
ВЫ — ЭКСПЕРТ-ОЦЕНЩИК КАЧЕСТВА СУММАРИЗАЦИЙ.
Пожалуйста, оцените целым числом качество следующих суммаризаций текста (генераций), используя шкалу Ликерта от 1 до 10, где 1 означает "очень плохая суммаризация", а 10 означает "отличная суммаризация".

Оценка должна учитывать:
- сохранение смысла исходного текста,
- степень сокращения текста (саммари не должно быть длиннее исходного текста),
- точность переданных фактов,
- логическую связность (нет лишнего),
- полноту изложения.

Также, пожалуйста, обоснуйте каждую оценку, избегая любой потенциальной предвзятости и гарантируя, что порядок, в котором были представлены ответы, не повлияет на ваше суждение.
Для каждой генерации необходимо вернуть:
- целочисленную оценку от 1 до 10,
- краткий комментарий.

Ответ верните строго в следующем формате JSON:

{
  "evaluations": [
    {
      "score": 8,
      "commentary": "Саммари точно передает основную мысль, краткое и без ошибок."
    },
    {
      "score": 5,
      "commentary": "Присутствуют фактические неточности и повторы."
    }
  ]
}

Количество элементов в списке "evaluations" должно точно соответствовать количеству представленных генераций. Убедитесь, что ключи score и commentary присутствуют для каждой генерации.

ИСХОДНЫЙ ТЕКСТ:
$input_text

ГЕНЕРАЦИИ ДЛЯ ОЦЕНКИ:
$summaries_block
""")


def load_data(paths):
    grouped = defaultdict(lambda: {"summaries": []})
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for item in json.load(f):
                id = item.get('id')
                assert id is not None, f"Missing id in {path}"
                grouped[id]["input_text"] = item['input_text']
                grouped[id]["reference_summary"] = item['reference_summary']
                grouped[id]["summaries"].append({
                    "source_file": os.path.basename(path),
                    "predicted_text": item['predicted_summary']
                })
    return grouped


def query_model(model, input_text, reference_summary, summaries):
    all_summaries = [{"predicted_text": reference_summary}] + summaries
    summaries_block = "\n\n".join(
        f"Генерация {i + 1}: \n{s['predicted_text']}" for i, s in enumerate(all_summaries)
    )
    prompt = PROMPT_TEMPLATE.substitute(input_text=input_text, summaries_block=summaries_block)

    for attempt in range(MAX_RETRIES):
        try:
            response = model.generate_content(
                [prompt],
                generation_config=GENERATION_CONFIG,
                safety_settings=SAFETY_SETTINGS,
                request_options={'timeout': 120}
            )
            result = json.loads(response.text)
            if len(result.get("evaluations", [])) == len(all_summaries):
                return result["evaluations"]
        except Exception:
            time.sleep(RETRY_DELAY_S * (attempt + 1))

    print("Failed after retries")
    return [{"score": -1, "commentary": "FAILED_AFTER_RETRIES"} for _ in all_summaries]


def save_results(grouped_results, out_dir, input_paths):
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "evaluation_results_comparative.jsonl"), 'w', encoding='utf-8') as f:
        for group in grouped_results:
            f.write(json.dumps(group, ensure_ascii=False) + '\n')

    by_file = defaultdict(list)
    for group in grouped_results:
        for s in group["summaries"]:
            if 1 <= s.get("evaluation_score", -1) <= 10:
                by_file[s["source_file"]].append(s["evaluation_score"])

    stats = {}
    for path in input_paths:
        fname = os.path.basename(path)
        scores = by_file[fname]
        if scores:
            stats[fname] = {
                "total_valid_samples": len(scores),
                "mean_score": round(np.mean(scores), 3),
                "std_deviation": round(np.std(scores), 3)
            }

    if "REFERENCE" in by_file:
        ref_scores = by_file["REFERENCE"]
        stats["REFERENCE"] = {
            "total_valid_samples": len(ref_scores),
            "mean_score": round(np.mean(ref_scores), 3),
            "std_deviation": round(np.std(ref_scores), 3)
        }

    with open(os.path.join(out_dir, "evaluation_stats_report.json"), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)


def main(input_files, output_dir):
    if not API_KEY:
        raise EnvironmentError("GOOGLE_API_KEY is not set.")

    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    grouped = load_data(input_files)

    grouped_results = []
    for id, item in tqdm(sorted(grouped.items()), desc="Evaluating"):
        evaluations = query_model(model, item['input_text'], item['reference_summary'], item['summaries'])

        entry = {
            "id": id,
            "input_text": item['input_text'],
            "summaries": []
        }

        for i, ev in enumerate(evaluations):
            if i == 0:
                source_file = "REFERENCE"
                predicted_text = item['reference_summary']
            else:
                source_file = item['summaries'][i - 1]['source_file']
                predicted_text = item['summaries'][i - 1]['predicted_text']

            entry["summaries"].append({
                "source_file": source_file,
                "predicted_summary": predicted_text,
                "evaluation_score": ev["score"],
                "evaluation_commentary": ev["commentary"]
            })

        grouped_results.append(entry)
        time.sleep(DELAY_BETWEEN_REQUESTS)

    save_results(grouped_results, output_dir, input_files)


if __name__ == '__main__':
    input_files = [
        "eval/res_synth/output_natural_2.json",
        "eval/res_synth/output_synth.json",
        "eval/res_synth/output_combined.json",
        "eval/res_synth/output_synth_DPO.json",
    ]

    output_dir = "llm_eval_synth"

    main(input_files, output_dir)
