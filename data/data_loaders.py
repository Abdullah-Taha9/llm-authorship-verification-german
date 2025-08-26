#!/usr/bin/env python3
"""
Data loading utilities for authorship verification experiments.
"""

from typing import Dict, Any

# HF Amazon AV
from datasets import load_dataset

# Local MAC loader (gz JSONL pairs)
import mac_loader


# ---------- Amazon (HF) helpers ----------

def load_dataset_with_seed(language: str, max_samples: int, seed: int = 42):
    """Random sample from sobamchan/amazon-review-authorship-verification."""
    print(f"[amazon/load_dataset] lang={language} max_samples={max_samples} seed={seed}")

    lang_map = {"english": "en", "german": "de"}
    dataset_lang = lang_map.get(language, language)

    ds = load_dataset(
        "sobamchan/amazon-review-authorship-verification",
        dataset_lang,
        split="train+validation+test"
    )
    print(f"[amazon/load_dataset] full_len={len(ds)}")

    ds = ds.shuffle(seed=seed)
    n = min(int(max_samples), len(ds))
    ds = ds.select(range(n))
    print(f"[amazon/load_dataset] returning n={len(ds)}")

    ds = ds.add_column("orig_idx", list(range(len(ds))))
    return ds


def load_amazon_topN_longest(language: str, top_n: int):
    lang_map = {"english": "en", "german": "de"}
    dataset_lang = lang_map.get(language, language)

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
    n = min(top_n, len(ds))
    ds_top = ds.select(range(len(ds) - n, len(ds)))
    ds_top = ds_top.add_column("orig_idx", list(range(len(ds_top))))

    print(f"[amazon/longest] top={n}")
    return ds_top


def load_amazon_topN_shortest(language: str, bottom_n: int):
    lang_map = {"english": "en", "german": "de"}
    dataset_lang = lang_map.get(language, language)

    ds = load_dataset(
        "sobamchan/amazon-review-authorship-verification",
        dataset_lang,
        split="train+validation+test"
    ).map(lambda ex, idx: {"orig_idx": idx}, with_indices=True)

    def _add_max_len(ex):
        t1 = ex["review_1"]["review_body"] or ""
        t2 = ex["review_2"]["review_body"] or ""
        ex["max_len"] = max(len(t1), len(t2))
        return ex

    ds = ds.map(_add_max_len, desc=f"Computing max_len for {language}")
    ds = ds.sort("max_len")
    n = min(bottom_n, len(ds))
    ds_short = ds.select(range(n))

    print(f"[amazon/shortest] n={n}")
    return ds_short


def load_amazon_topN_longest_balanced(language: str, top_n: int):
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
            keep_indices.append(i); pos_count += 1
        elif (not is_pos) and neg_count < neg_target:
            keep_indices.append(i); neg_count += 1
        if pos_count >= pos_target and neg_count >= neg_target:
            break

    selected = ds_desc.select(keep_indices)
    selected = selected.add_column("orig_idx", list(range(len(selected))))

    print(f"[amazon/longest_balanced] pos={pos_count} neg={neg_count} total={len(selected)}")
    return selected


def load_amazon_topN_shortest_balanced(language: str, bottom_n: int):
    lang_map = {"english": "en", "german": "de"}
    dataset_lang = lang_map.get(language, language)

    pos_target = bottom_n // 2
    neg_target = bottom_n - pos_target

    ds = load_dataset(
        "sobamchan/amazon-review-authorship-verification",
        dataset_lang,
        split="train+validation+test"
    ).map(lambda ex, idx: {"orig_idx": idx}, with_indices=True)

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
            keep_indices.append(i); pos_count += 1
        elif (not is_pos) and neg_count < neg_target:
            keep_indices.append(i); neg_count += 1
        if pos_count >= pos_target and neg_count >= neg_target:
            break

    selected = ds_sorted.select(keep_indices)
    print(f"[amazon/shortest_balanced] pos={pos_count} neg={neg_count} total={len(selected)}")
    return selected


# ---------- MAC helpers ----------

def load_mac_pairs(root_dir: str,
                   mac_language: str,
                   split: str = "train",
                   max_samples: int = 500,
                   seed: int = 42):
    """
    Return a Python list of dicts with keys:
      review_1.review_body, review_2.review_body, label, orig_idx
    compatible with the runner loop.
    """
    lang_map = {
        "german":  "german/de_wikipedia",
        "english": "english/en_wikipedia",
        # allow short codes too:
        "de": "german/de_wikipedia",
        "en": "english/en_wikipedia",
    }
    lang_dir = lang_map.get(mac_language, mac_language)

    pairs = mac_loader.load_mac_split_as_pairs_unique_query(
        root_dir=root_dir,
        lang_dir=lang_dir,
        split=split,
        max_samples=max_samples,
        seed=seed
    )
    # mac_loader already sets orig_idx; ensure compatibility
    for i, ex in enumerate(pairs):
        if "orig_idx" not in ex:
            ex["orig_idx"] = i
    print(f"[mac/load] lang_dir={lang_dir} split={split} n={len(pairs)}")
    return pairs


# ---------- Main selector ----------

def get_data_loader(config: Dict[str, Any]):
    """
    Choose dataset by config:
      - dataset: "amazon_review" | "mac"
      - dataset_language: "english" | "german" | ("en"/"de" also accepted for MAC)
      - data_loader: one of {random,longest,shortest,longest_balanced,shortest_balanced} for Amazon
      - For MAC: provide mac_root_dir (default "wikipedia_1M_author_corpus_v1.2") and mac_split ("train"/"dev"/"test")
    """
    dataset = config["dataset"]
    language = config["dataset_language"]
    max_samples = int(config["max_samples"])
    seed = int(config.get("random_seed", 42))

    if dataset == "amazon_review":
        dl = config.get("data_loader", "random")
        if dl in (None, "random"):
            return load_dataset_with_seed(language, max_samples, seed)
        elif dl == "longest":
            return load_amazon_topN_longest(language, max_samples)
        elif dl == "shortest":
            return load_amazon_topN_shortest(language, max_samples)
        elif dl == "longest_balanced":
            return load_amazon_topN_longest_balanced(language, max_samples)
        elif dl == "shortest_balanced":
            return load_amazon_topN_shortest_balanced(language, max_samples)
        else:
            raise ValueError(f"Unknown data_loader: {dl}")

    elif dataset == "mac":
        root_dir = config.get("mac_root_dir", "wikipedia_1M_author_corpus_v1.2")
        split = config.get("mac_split", "train")  # "train" | "dev" | "test"
        return load_mac_pairs(root_dir, language, split, max_samples, seed)

    else:
        raise ValueError(f"Unknown dataset: {dataset}")
