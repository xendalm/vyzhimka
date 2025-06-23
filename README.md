# vyzhimka

This repository contains code and experiments for evaluating the impact of training methods and synthetic data on the quality of Russian abstractive text summarization.

## Training and Evaluation overview

**Models:**
- FRED-T5-large (primary generation model, used for fine-tuning on summarization task and DPO)
- MT5-large (SEAHORSE reward model used for scoring summaries in DPO alignment)

**Data:**
- Open-source Russian summarization datasets: MLSUM, Gazeta, XLSum, WikiLingua, etc.
- [Synthetic data](https://huggingface.co/datasets/xendalm/taiga-sum) generated using Gemini 2.0 (based on Taiga/N+1 and filtered Fontanka)

**Evaluation:**
- Standard automatic metrics
- LLM-as-a-judge
- Natural Language Inference to assess factual consistency

## Structure

- `model` — code for summarization model SFT and DPO training 
- `eval` — code for evaluation metrics and LLM-as-a-judge evaluation
- `notebooks/datasets.ipynb` — data processing scripts
- `seahorse_metric` — code for training, validation and evaluation of SEAHORSE reward model

## Results

### Automatic Metrics (Natural Test Set)

| Model                       | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU  | METEOR | ChrF++ | BERTScore-F1 | LaBSE | Compression |
|-----------------------------|---------|---------|---------|-------|--------|--------|---------------|--------|---------|
| SFT-natural                 | 0.193   | 0.093   | 0.189   | 0.100 | 0.236  | 32.97  | 0.731         | 0.762  | 3.28 ± 1.35 |
| SFT-synth                   | 0.332   | 0.169   | 0.319   | 0.155 | 0.393  | 49.52  | 0.781         | 0.868  | 1.81 ± 0.65 |
| SFT-combined                | 0.300   | 0.160   | 0.288   | 0.146 | 0.413  | 51.23  | 0.770         | 0.862  | 1.47 ± 0.48 |
| SFT-DPO                     | 0.349   | 0.184   | 0.337   | 0.162 | 0.396  | 49.32  | 0.785         | 0.869  | 1.90 ± 0.65 |
| Baseline (gemini-2.0-flash) | 0.329   | 0.194   | 0.321   | 0.163 | 0.361  | 44.87  | 0.794         | 0.878  | 2.71 ± 1.03 |
| Reference                   | 1.000   | 1.000   | 1.000   | 1.000 | 1.000  | 100.0  | 1.000         | 1.000  | 2.58 ± 1.12 |

> SFT-natural demonstrated the lowest performance. SFT-synth showed significant improvement, confirming the quality of the synthetic dataset. DPO improved all metrics while reducing output length, indicating better conciseness.

### Automatic Metrics (Synthetic Test Set)

| Model         | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU  | METEOR | ChrF++ | BERTScore-F1 | LaBSE | Compression |
|---------------|---------|---------|---------|-------|--------|--------|---------------|--------|-------------|
| SFT-natural   | 0.261   | 0.104   | 0.252   | 0.090 | 0.267  | 36.57  | 0.744         | 0.792  | 5.04 ± 2.04 |
| SFT-synth     | 0.343   | 0.159   | 0.330   | 0.111 | 0.414  | 48.81  | 0.770         | 0.859  | 2.20 ± 0.66 |
| SFT-combined  | 0.348   | 0.160   | 0.334   | 0.102 | 0.420  | 48.88  | 0.765         | 0.860  | 1.87 ± 0.53 |
| SFT-DPO       | 0.358   | 0.162   | 0.345   | 0.113 | 0.416  | 49.16  | 0.774         | 0.861  | 2.26 ± 0.68 |
| Reference     | 1.000   | 1.000   | 1.000   | 1.000 | 1.000  | 100.0  | 1.000         | 1.000  | 4.68 ± 1.75 |

---

### Factual Consistency (SummaC-ZS)

| Model         | SummaC-ZS (Natural) | SummaC-ZS (Synthetic) |
|---------------|----------------------|-------------------------|
| SFT-natural   | 0.644 ± 0.178        | 0.575 ± 0.194           |
| SFT-synth     | 0.536 ± 0.213        | 0.515 ± 0.152           |
| SFT-combined  | 0.560 ± 0.186        | 0.501 ± 0.143           |
| SFT-DPO       | 0.534 ± 0.213        | 0.512 ± 0.162           |
| Reference     | 0.571 ± 0.235        | 0.435 ± 0.209           |

[//]: # (| Baseline      | 0.498 ± 0.269        | —                       |)

> The highest factual consistency score was achieved by the SFT-natural model, likely due to its tendency to copy source text. Despite slightly lower scores, SFT-DPO eliminated hallucinations and improved factual quality not always captured by the metric.

---

### Copying Degree

| Model         | Word Cont. (Natural) | 3-gram Cont. (Natural) | Word Cont. (Synthetic) | 3-gram Cont. (Synthetic) |
|---------------|----------------------|-------------------------|-------------------------|---------------------------|
| SFT-natural   | 0.795 ± 0.231        | 0.567 ± 0.322           | 0.872 ± 0.105           | 0.585 ± 0.201             |
| SFT-synth     | 0.641 ± 0.108        | 0.213 ± 0.120           | 0.631 ± 0.077           | 0.187 ± 0.070             |
| SFT-combined  | 0.704 ± 0.126        | 0.364 ± 0.155           | 0.685 ± 0.080           | 0.262 ± 0.077             |
| SFT-DPO       | 0.658 ± 0.111        | 0.220 ± 0.118           | 0.642 ± 0.076           | 0.191 ± 0.072             |
| Reference     | 0.627 ± 0.142        | 0.196 ± 0.134           | 0.621 ± 0.098           | 0.126 ± 0.074             |

> Copying was clearly higher for SFT-natural. Models trained on synthetic data produced significantly more abstractive outputs, approaching the level of human-written summaries.

---

### LLM-as-a-Judge Score

| Model         | Avg. Score (Natural) | Avg. Score (Synthetic) |
|---------------|----------------------|--------------------------|
| SFT-natural   | 4.67 ± 2.24          | 6.12 ± 1.35              |
| SFT-synth     | 6.32 ± 1.82          | 5.62 ± 1.97              |
| SFT-combined  | 5.73 ± 2.05          | 5.63 ± 1.96              |
| SFT-DPO       | 6.97 ± 1.71          | 6.32 ± 1.83              |
| Reference     | 8.53 ± 0.64          | 8.85 ± 0.36              |

[//]: # (| Baseline      | 8.83 ± 0.57          | —                        |)

> SFT-DPO received the highest rating among trained models. 

[//]: # (> Slight advantage of Gemini baseline may reflect alignment with its own style. The highest scores for the Gemini baseline can be attributed to the known LLM judges bias towards their own predictions.)

[//]: # (---)

[//]: # ()
[//]: # (### SEAHORSE-Q6 &#40;Conciseness Estimator&#41;)

[//]: # ()
[//]: # (| Model         | Score &#40;Natural&#41; |)

[//]: # (|---------------|------------------|)

[//]: # (| SFT-natural   | 0.543 ± 0.183    |)

[//]: # (| SFT-synth     | 0.517 ± 0.189    |)

[//]: # (| SFT-combined  | 0.458 ± 0.195    |)

[//]: # (| SFT-DPO       | 0.561 ± 0.167    |)

[//]: # (| Baseline      | 0.633 ± 0.142    |)

[//]: # (| Reference     | 0.647 ± 0.143    |)

[//]: # ()
[//]: # (> SEAHORSE-Q6 scores consistently increased towards SFT-DPO. The overall trend confirms improvement in conciseness and quality, with reference summaries and Gemini scoring highest as expected.)

[//]: # (---)
