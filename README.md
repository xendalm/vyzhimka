# vyzhimka

This repository contains code and experiments for evaluating the impact of training methods and synthetic data on the quality of Russian abstractive text summarization.

# Training and Evaluation overview

Models:
- FRED-T5-large (primary generation model, used for fine-tuning on summarization task and DPO)
- MT5-large (SEAHORSE reward model used for scoring summaries in DPO alignment)

Data:
- Open-source Russian summarization datasets (MLSUM, Gazeta, XLSum, WikiLingua, etc.)
- Synthetic data generated using Gemini 2.0 (based on Taiga/N+1 and filtered Fontanka). This dataset was designed to provide high-quality examples.

Evaluation:
- Standard auto metrics (Rouge, BLEU, BERTScore, etc.)
- LLM-as-a-judge
- Natural Language Inference (used to assess the factual consistency)

# Structure

- `model` - code for summarization model SFT and DPO training 
- `eval` - code for evaluation metrics and LLM-as-a-judge evaluation
- `notebooks/datasets.ipynb` - data processing scripts
- `seahorse_metric` - code for training, validation and evaluation of SEAHORSE reward model

# Results
