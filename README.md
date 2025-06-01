# rusumm

This repository contains code and experiments for evaluating the impact of synthetic data and training methods on the quality of Russian abstractive text summarization.

# Training and Evaluation overview

Models:
- FRED-T5-large (used for fine-tuning on summarization task and DPO)
- MT5-large (SEAHORSE reward model used to score summaries for use in DPO alignment)

Data:
- Open-source Russian summarization datasets (MLSUM, Gazeta, XLSum, WikiLingua, etc.)
- Synthetic data generated using Gemini 2.0 (based on Taiga/N+1 and filtered Fontanka)

Evaluation:
- Standard auto-evaluation metrics (Rouge, BLEU, BERTScore, etc.)
- LLM-as-a-judge evaluation
- Natural Language Inference evaluation

# Structure

- `model` - code for summarization model SFT and DPO training 
- `eval` - code for evaluation metrics and LLM-as-a-judge evaluation
- `notebooks/datasets.ipynb` - data processing scripts
- `seahorse_metric` - code for training, validation and evaluation of SEAHORSE reward model

# Results
