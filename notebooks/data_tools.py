from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRED-T5-large")

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 300
TASK_PROMPT = "<LM> Сократи текст: "

def is_short_enough(example):
    text_tokens = tokenizer(
        TASK_PROMPT + example["text"],
        truncation=False,
    )["input_ids"]

    summary_tokens = tokenizer(
        example["summary"],
        truncation=False,
    )["input_ids"]

    return len(text_tokens) < MAX_INPUT_LENGTH and len(summary_tokens) < MAX_TARGET_LENGTH

def count_truncations(example):
    input_tokens = tokenizer(TASK_PROMPT + example["text"])["input_ids"]
    target_tokens = tokenizer(example["summary"])["input_ids"]
    return {
        "input_truncated": len(input_tokens) > MAX_INPUT_LENGTH,
        "target_truncated": len(target_tokens) > MAX_TARGET_LENGTH
    }

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("ai-forever/FRED-T5-large")

MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 300
TASK_PROMPT = "<LM> Сократи текст: "

def is_short_enough(example):
    text_tokens = tokenizer(
        TASK_PROMPT + example["text"],
        truncation=False,
    )["input_ids"]

    summary_tokens = tokenizer(
        example["summary"],
        truncation=False,
    )["input_ids"]

    return len(text_tokens) < MAX_INPUT_LENGTH and len(summary_tokens) < MAX_TARGET_LENGTH

from nltk.tokenize import word_tokenize

def count(example):
    input_tokens = tokenizer(TASK_PROMPT + example["text"])["input_ids"]
    target_tokens = tokenizer(example["summary"])["input_ids"]
    input_words = len([token for token in word_tokenize(example['text'], language='russian') if token.isalpha()])
    target_words = len([token for token in word_tokenize(example['summary'], language='russian') if token.isalpha()])
    return {
        "input_tokens": len(input_tokens),
        "target_tokens": len(target_tokens),
        "input_words": input_words,
        "target_words": target_words,
        "compression_ratio_words": input_words / target_words,
        "compression_ratio_tokens": len(input_tokens) / len(target_tokens),
    }

import matplotlib.pyplot as plt
import math
import numpy as np

def plot_histograms(df, add_fields=None):
    fields = [
        "input_tokens", 
        "input_words", 
        "compression_ratio_words", 
        "target_tokens", 
        "target_words", 
        "compression_ratio_tokens"
    ]
    if add_fields:
        fields.extend(add_fields)
    
    n = len(fields)
    n_cols = 3
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for i, field in enumerate(fields):
        data = df[field]
        data = data[np.isfinite(data)]

        axes[i].hist(data, bins=50, color='skyblue', edgecolor='black')
        axes[i].set_title(field)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    # Удаление неиспользуемых осей
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def cmp_train_test(train, test):
    columns = [
        "compression_ratio_words",
        "compression_ratio_tokens",
        "input_tokens",
        "target_tokens",
        "input_words",
        "target_words",
        "score"
    ]
    n_rows = (len(columns) + 1) // 2
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
    axes = axes.flatten()
    for idx, col in enumerate(columns):
        sns.kdeplot(train[col], label="train", ax=axes[idx], fill=True)
        sns.kdeplot(test[col], label="test", ax=axes[idx], fill=True)
        axes[idx].set_title(f"KDE of {col}")
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel("Density")
        axes[idx].legend()
        axes[idx].grid(True)
    fig.suptitle("Empirical Densities for Train vs Test Distributions", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

import re
from nltk.tokenize import sent_tokenize

link_pattern = re.compile(r'https?://\S+|www\.\S+', re.IGNORECASE)

def remove_link_sentences(text):
    if text is None:
        return text, 0
    sentences = sent_tokenize(text, language='russian')
    clean_sentences = [s for s in sentences if not link_pattern.search(s)]
    changed = int(len(clean_sentences) < len(sentences))
    return ' '.join(clean_sentences), changed