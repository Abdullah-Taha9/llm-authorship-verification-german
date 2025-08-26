# data/mac_loader.py
import gzip, json, random
from pathlib import Path
from collections import defaultdict

def _as_authorset(x):
    if x is None:
        return set()
    if isinstance(x, (list, tuple)):
        return {str(a) for a in x if a is not None}
    return {str(x)}

def _read_side(path):
    items = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            doc_id  = obj.get("documentID")
            text    = obj.get("fullText")
            authors = _as_authorset(obj.get("authorIDs"))
            if not doc_id or not isinstance(text, str) or not text.strip():
                continue
            items.append({"doc_id": str(doc_id), "authors": authors, "text": text})
    return items

def load_mac_split_as_pairs_unique_query(
    root_dir,
    lang_dir,
    split="train",
    max_samples=500,
    seed=42,
    prefer_positive_first=True,
    max_trials_per_query=100
):
    base = Path(root_dir) / lang_dir
    q_path = base / f"{split}_queries.jsonl.gz"
    c_path = base / f"{split}_candidates.jsonl.gz"
    if not q_path.exists() or not c_path.exists():
        raise FileNotFoundError(
            f"MAC files not found. Expected:\n  {q_path}\n  {c_path}\n"
            "Check mac_root_dir/lang_dir/split in your config."
        )

    rng = random.Random(seed)
    queries = _read_side(q_path)
    cands   = _read_side(c_path)

    author_to_cand_idx = defaultdict(list)
    for idx, c in enumerate(cands):
        for a in c["authors"]:
            author_to_cand_idx[a].append(idx)
    all_cand_indices = list(range(len(cands)))

    rng.shuffle(queries)
    seek_positive = bool(prefer_positive_first)

    pairs = []
    n_pos = n_neg = 0

    for q in queries:
        if max_samples and len(pairs) >= max_samples:
            break

        qa = q["authors"]
        made = False

        if seek_positive:
            pool = []
            for a in qa:
                pool.extend(author_to_cand_idx.get(a, []))
            pool = [ci for ci in set(pool) if cands[ci]["doc_id"] != q["doc_id"]]
            rng.shuffle(pool)
            if pool:
                ci = pool[0]
                pairs.append({
                    "orig_idx": len(pairs),
                    "review_1": {"review_body": q["text"]},
                    "review_2": {"review_body": cands[ci]["text"]},
                    "label": True
                })
                n_pos += 1
                made = True
        else:
            tried = 0
            while tried < max_trials_per_query:
                ci = rng.choice(all_cand_indices)
                c = cands[ci]
                if c["doc_id"] == q["doc_id"]:
                    tried += 1; continue
                if qa.isdisjoint(c["authors"]):
                    pairs.append({
                        "orig_idx": len(pairs),
                        "review_1": {"review_body": q["text"]},
                        "review_2": {"review_body": c["text"]},
                        "label": False
                    })
                    n_neg += 1
                    made = True
                    break
                tried += 1

        if made:
            seek_positive = not seek_positive

    print(f"Loaded {len(pairs)} pairs (unique query) from {lang_dir} [{split}]")
    print(f"Positives: {n_pos} | Negatives: {n_neg} | Total queries used: {len(pairs)}")
    return pairs
