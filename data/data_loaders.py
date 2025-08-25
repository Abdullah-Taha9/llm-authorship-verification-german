#!/usr/bin/env python3
"""
Data loading utilities for authorship verification experiments.
Separated from main experiment code for better organization.
"""

from datasets import load_dataset
from typing import Dict, Any


def load_dataset_with_seed(language: str, max_samples: int, seed: int = 42):
    """Load dataset with specified parameters using random sampling."""
    print(f"[load_dataset] lang={language} max_samples={max_samples} seed={seed}")
    
    # Map language configuration to dataset language codes
    lang_map = {"english": "en", "german": "de"}
    dataset_lang = lang_map.get(language, language)
    
    ds = load_dataset(
        "sobamchan/amazon-review-authorship-verification",
        dataset_lang,
        split="train+validation+test"
    )
    print(f"[load_dataset] full_len={len(ds)}")
    
    ds = ds.shuffle(seed=seed)
    n = min(int(max_samples), len(ds))
    ds = ds.select(range(n))
    print(f"[load_dataset] returning n={len(ds)}")
    
    ds = ds.add_column("orig_idx", list(range(len(ds))))
    return ds


def load_amazon_topN_longest(language: str, top_n: int):
    """Load top N longest pairs by min_len."""
    lang_map = {"english": "en", "german": "de"}
    dataset_lang = lang_map.get(language, language)
    
    ds = load_dataset(
        "sobamchan/amazon-review-authorship-verification",
        dataset_lang,
        split="train+validation+test"
    )

    def _add_min_len(ex):
        t1 = ex["review_1"]["review_body"]
        t2 = ex["review_2"]["review_body"]
        ex["min_len"] = min(len(t1) if t1 else 0, len(t2) if t2 else 0)
        return ex

    ds = ds.map(_add_min_len, desc=f"Computing min_len for {language}")
    ds = ds.sort("min_len")
    n = min(top_n, len(ds))
    ds_top = ds.select(range(len(ds) - n, len(ds)))
    ds_top = ds_top.add_column("orig_idx", list(range(len(ds_top))))

    print(f"Loaded top-{n} longest pairs for {language} (by min_len).")
    return ds_top


def load_amazon_topN_shortest(language: str, bottom_n: int):
    """Load bottom N shortest pairs by max_len."""
    lang_map = {"english": "en", "german": "de"}
    dataset_lang = lang_map.get(language, language)
    
    ds = load_dataset(
        "sobamchan/amazon-review-authorship-verification",
        dataset_lang,
        split="train+validation+test"
    )
    
    ds = ds.map(lambda ex, idx: {"orig_idx": idx}, with_indices=True)

    def _add_max_len(ex):
        t1 = ex["review_1"]["review_body"] or ""
        t2 = ex["review_2"]["review_body"] or ""
        ex["max_len"] = max(len(t1), len(t2))
        return ex

    ds = ds.map(_add_max_len, desc=f"Computing max_len for {language}")
    ds = ds.sort("max_len")
    n = min(bottom_n, len(ds))
    ds_short = ds.select(range(n))

    print(f"Loaded {n} shortest pairs for {language} (by max_len).")
    return ds_short


def load_amazon_topN_longest_balanced(language: str, top_n: int):
    """Load top N longest pairs with balanced classes."""
    lang_map = {"english": "en", "german": "de"}
    dataset_lang = lang_map.get(language, language)
    
    pos_target = top_n // 2
    neg_target = top_n - pos_target

    ds = load_dataset(
        "sobamchan/amazon-review-authorship-verification",
        dataset_lang,
        split="train+validation+test"
    )

    def _add_min_len(ex):
        t1 = ex["review_1"]["review_body"] or ""
        t2 = ex["review_2"]["review_body"] or ""
        ex["min_len"] = min(len(t1), len(t2))
        return ex

    ds = ds.map(_add_min_len, desc=f"Computing min_len for {language}")
    ds = ds.sort("min_len")
    idx_desc = list(range(len(ds) - 1, -1, -1))
    ds_desc = ds.select(idx_desc)

    pos_count = neg_count = 0
    keep_indices = []
    for i, ex in enumerate(ds_desc):
        is_pos = bool(ex["label"])
        if is_pos and pos_count < pos_target:
            keep_indices.append(i)
            pos_count += 1
        elif (not is_pos) and neg_count < neg_target:
            keep_indices.append(i)
            neg_count += 1
        if pos_count >= pos_target and neg_count >= neg_target:
            break

    selected = ds_desc.select(keep_indices)
    selected = selected.add_column("orig_idx", list(range(len(selected))))

    print(f"Selected {len(selected)} for {language}: pos={pos_count}, neg={neg_count}")
    return selected


def load_amazon_topN_shortest_balanced(language: str, bottom_n: int):
    """Load bottom N shortest pairs with balanced classes."""
    lang_map = {"english": "en", "german": "de"}
    dataset_lang = lang_map.get(language, language)
    
    pos_target = bottom_n // 2
    neg_target = bottom_n - pos_target

    ds = load_dataset(
        "sobamchan/amazon-review-authorship-verification",
        dataset_lang,
        split="train+validation+test"
    )
    
    ds = ds.map(lambda ex, idx: {"orig_idx": idx}, with_indices=True)

    def _add_max_len(ex):
        t1 = ex["review_1"]["review_body"] or ""
        t2 = ex["review_2"]["review_body"] or ""
        ex["max_len"] = max(len(t1), len(t2))
        return ex

    ds = ds.map(_add_max_len, desc=f"Computing max_len for {language}")
    ds_sorted = ds.sort("max_len")

    pos_count = neg_count = 0
    keep_indices = []
    for i, ex in enumerate(ds_sorted):
        is_pos = bool(ex["label"])
        if is_pos and pos_count < pos_target:
            keep_indices.append(i)
            pos_count += 1
        elif (not is_pos) and neg_count < neg_target:
            keep_indices.append(i)
            neg_count += 1
        if pos_count >= pos_target and neg_count >= neg_target:
            break

    selected = ds_sorted.select(keep_indices)
    print(f"Selected {len(selected)} shortest for {language}: pos={pos_count}, neg={neg_count}")
    return selected


def get_data_loader(config: Dict[str, Any]):
    """Get the appropriate data loader based on configuration."""
    language = config["dataset_language"]
    max_samples = config["max_samples"]
    seed = config["random_seed"]
    data_loader = config.get("data_loader")
    
    if data_loader is None or data_loader == "random":
        return load_dataset_with_seed(language, max_samples, seed)
    elif data_loader == "longest":
        return load_amazon_topN_longest(language, max_samples)
    elif data_loader == "shortest":
        return load_amazon_topN_shortest(language, max_samples)
    elif data_loader == "longest_balanced":
        return load_amazon_topN_longest_balanced(language, max_samples)
    elif data_loader == "shortest_balanced":
        return load_amazon_topN_shortest_balanced(language, max_samples)
    else:
        raise ValueError(f"Unknown data_loader: {data_loader}")
