"""
candidate_miss_diagnostic.py

Diagnostic analysis of true matches missed by candidate_generation_fast.py
for the Amazon ML Challenge 2026 Business Entity Resolution project.

Analyzes:
1. Ground truth recovery vs candidate pairs on the exact 1000 S1 validation sample.
2. Token overlap, numeric address overlap, script divergence, and missingness.
3. Simulation of alternative blocking rules to recover missed pairs.
"""

from __future__ import annotations

import os
import sys
import gc
import json
import time
import ast
import random
import unicodedata
import re
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple
from collections import defaultdict

import pandas as pd

# Add source directory to path to enable direct import of preprocessing.py
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import preprocessing as prep_mod
except ImportError:
    prep_mod = None


# --- Path Resolution ---
def resolve_project_root() -> Path:
    candidates = [
        Path(__file__).resolve().parents[3],
        Path(__file__).resolve().parents[2],
        Path.cwd(),
    ]
    for c in candidates:
        if (c / "dataset" / "train").exists():
            return c
    return candidates[0]


PROJECT_ROOT = resolve_project_root()
DATASET_TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAIN_SOURCE_FILES = {
    "source1": DATASET_TRAIN_DIR / "train_source1.tsv",
    "source2": DATASET_TRAIN_DIR / "train_source2.tsv",
    "source3": DATASET_TRAIN_DIR / "train_source3.tsv",
}

GROUND_TRUTH_FILE = DATASET_TRAIN_DIR / "train_ground_truth.tsv"
OUTPUT_PAIRS_FILE = OUTPUT_DIR / "candidate_pairs_fast.tsv"
DIAGNOSTIC_TSV_FILE = OUTPUT_DIR / "candidate_miss_diagnostic.tsv"
DIAGNOSTIC_REPORT_FILE = OUTPUT_DIR / "candidate_miss_diagnostic_report.json"

CHUNKSIZE = 200_000


# --- Normalization & Tokenization Utilities ---
def _default_normalize(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return ""
    text_str = unicodedata.normalize("NFKC", str(text))
    text_str = re.sub(r"\s+", " ", text_str).strip().casefold()
    return text_str


def normalize_name(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return ""
    if prep_mod is not None:
        for fn_name in ("normalize_business_name", "normalize_name", "normalize_text", "clean_text"):
            if hasattr(prep_mod, fn_name):
                fn = getattr(prep_mod, fn_name)
                res = fn(text)
                return "" if res is None or pd.isna(res) else str(res).strip()
    return _default_normalize(text)


def normalize_address(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return ""
    if prep_mod is not None:
        for fn_name in ("normalize_business_address", "normalize_address", "normalize_text", "clean_text"):
            if hasattr(prep_mod, fn_name):
                fn = getattr(prep_mod, fn_name)
                res = fn(text)
                return "" if res is None or pd.isna(res) else str(res).strip()
    return _default_normalize(text)


def normalize_country(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return ""
    if prep_mod is not None:
        for fn_name in ("normalize_country", "clean_country", "normalize_text"):
            if hasattr(prep_mod, fn_name):
                fn = getattr(prep_mod, fn_name)
                res = fn(text)
                return "" if res is None or pd.isna(res) else str(res).strip()
    return _default_normalize(text)


def safe_parse_tokens(raw_val: any, norm_val: str) -> Set[str]:
    """Extracts valid tokens >= 2 chars, reusing preprocessed column if available."""
    if pd.notna(raw_val) and str(raw_val).strip():
        val_str = str(raw_val).strip()
        if val_str.startswith('['):
            try:
                parsed = ast.literal_eval(val_str)
                return {str(x).strip().lower() for x in parsed if len(str(x).strip()) >= 2}
            except Exception:
                pass
        return {t.strip().lower() for t in val_str.replace(',', ' ').split() if len(t.strip()) >= 2}

    if norm_val:
        return {t for t in re.split(r'\W+', str(norm_val).lower()) if len(t) >= 2}
    return set()


def extract_numeric_tokens(addr_text: str) -> Set[str]:
    if not addr_text:
        return set()
    return set(re.findall(r"\d+", addr_text))


def detect_primary_script(text: str) -> str:
    """Detects primary Unicode script category for alphabetic characters."""
    if not text:
        return "UNKNOWN"
    counts = defaultdict(int)
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            prefix = name.split()[0] if name else "UNKNOWN"
            counts[prefix] += 1

    if not counts:
        return "UNKNOWN"
    return max(counts.items(), key=lambda item: item[1])[0]


def compare_scripts(name1: str, name2: str) -> str:
    s1 = detect_primary_script(name1)
    s2 = detect_primary_script(name2)
    if s1 == "UNKNOWN" or s2 == "UNKNOWN":
        return "UNKNOWN"
    return "SAME" if s1 == s2 else "DIFFERENT"


# --- Data Structures ---
class RecordInfo:
    __slots__ = (
        "entity_id",
        "country",
        "name_raw",
        "name_norm",
        "addr_raw",
        "addr_norm",
        "name_tokens",
        "addr_tokens",
        "numeric_tokens",
    )

    def __init__(
        self,
        entity_id: str,
        country: str,
        name_raw: str,
        name_norm: str,
        addr_raw: str,
        addr_norm: str,
        name_tokens: Set[str],
        addr_tokens: Set[str],
        numeric_tokens: Set[str],
    ):
        self.entity_id = entity_id
        self.country = country
        self.name_raw = name_raw
        self.name_norm = name_norm
        self.addr_raw = addr_raw
        self.addr_norm = addr_norm
        self.name_tokens = name_tokens
        self.addr_tokens = addr_tokens
        self.numeric_tokens = numeric_tokens


def extract_record_from_row(row: pd.Series) -> RecordInfo:
    eid = str(row.get("entity_id", "")).strip()

    raw_country = row.get("country", "")
    country_norm = row.get("country_normalized")
    if pd.isna(country_norm) or not country_norm:
        country_norm = normalize_country(raw_country)
    country = str(country_norm).strip()

    name_raw = str(row.get("business_name", "")).strip() if pd.notna(row.get("business_name")) else ""
    name_norm = row.get("business_name_normalized")
    if pd.isna(name_norm) or not name_norm:
        name_norm = normalize_name(name_raw)
    name_norm = str(name_norm).strip()

    addr_raw = str(row.get("business_address", "")).strip() if pd.notna(row.get("business_address")) else ""
    addr_norm = row.get("business_address_normalized")
    if pd.isna(addr_norm) or not addr_norm:
        addr_norm = normalize_address(addr_raw)
    addr_norm = str(addr_norm).strip()

    n_toks = safe_parse_tokens(row.get("business_name_tokens"), name_norm)
    a_toks = safe_parse_tokens(row.get("business_address_tokens"), addr_norm)
    num_toks = extract_numeric_tokens(addr_raw or addr_norm)

    return RecordInfo(
        entity_id=eid,
        country=country,
        name_raw=name_raw,
        name_norm=name_norm,
        addr_raw=addr_raw,
        addr_norm=addr_norm,
        name_tokens=n_toks,
        addr_tokens=a_toks,
        numeric_tokens=num_toks,
    )


def get_1000_s1_sample(seed: int = 42) -> List[str]:
    """Reproduces the exact reservoir sample from candidate_generation_fast.py."""
    rng = random.Random(seed)
    reservoir = []
    n_seen = 0

    reader = pd.read_csv(
        TRAIN_SOURCE_FILES["source1"],
        sep="\t",
        usecols=["entity_id"],
        chunksize=CHUNKSIZE,
        dtype=str,
    )
    for chunk in reader:
        for eid in chunk["entity_id"]:
            if pd.isna(eid) or not str(eid).strip():
                continue
            eid_str = str(eid).strip()
            if len(reservoir) < 1000:
                reservoir.append(eid_str)
            else:
                j = rng.randint(0, n_seen)
                if j < 1000:
                    reservoir[j] = eid_str
            n_seen += 1

    return reservoir


def run_diagnostic():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    print("=" * 70)
    print("CANDIDATE MISS DIAGNOSTIC (Amazon ML Challenge 2026)")
    print("=" * 70)

    if not OUTPUT_PAIRS_FILE.exists():
        raise FileNotFoundError(
            f"Candidate pairs file not found: {OUTPUT_PAIRS_FILE}\n"
            "Please run candidate_generation_fast.py --mode sample_validation first."
        )

    t0 = time.time()

    # 1. Reproduce 1000 S1 IDs
    print("\n[Step 1/5] Reproducing exact 1000 S1 validation sample (seed=42)...")
    s1_sample_ids = get_1000_s1_sample(seed=42)
    s1_sample_set = set(s1_sample_ids)
    print(f"  Reproduced {len(s1_sample_set)} S1 entities.")

    # 2. Load ground truth for sample
    print("\n[Step 2/5] Reading ground truth for sampled entities...")
    true_pairs: Set[Tuple[str, str]] = set()
    gt_reader = pd.read_csv(
        GROUND_TRUTH_FILE,
        sep="\t",
        chunksize=CHUNKSIZE,
        dtype=str,
        keep_default_na=False,
    )
    for chunk in gt_reader:
        mask = chunk["source1_entity_id"].isin(s1_sample_set)
        if mask.any():
            for _, row in chunk[mask].iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                raw_matched = str(row.get("matched_entity_ids", "")).strip()
                if raw_matched:
                    for cand_id in raw_matched.split(","):
                        cand_id = cand_id.strip()
                        if cand_id:
                            true_pairs.add((s1_id, cand_id))

    total_true_matches = len(true_pairs)
    print(f"  Total true match pairs in scope: {total_true_matches}")

    # 3. Load existing candidate pairs
    print(f"\n[Step 3/5] Loading existing candidate pairs from {OUTPUT_PAIRS_FILE.name}...")
    candidate_pairs: Set[Tuple[str, str]] = set()
    with open(OUTPUT_PAIRS_FILE, "r", encoding="utf-8") as f:
        header = next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                s1_id, cand_id = parts[0], parts[1]
                if s1_id in s1_sample_set:
                    candidate_pairs.add((s1_id, cand_id))

    recovered_pairs = true_pairs & candidate_pairs
    missed_pairs = true_pairs - candidate_pairs

    recovered_count = len(recovered_pairs)
    missed_count = len(missed_pairs)
    recall = recovered_count / total_true_matches if total_true_matches > 0 else 0.0

    print(f"  Recovered matches: {recovered_count} ({recall * 100:.2f}%)")
    print(f"  Missed matches:    {missed_count}")

    if missed_count == 0:
        print("\nNo true matches were missed! Diagnostic complete.")
        return

    # 4. Stream Source Records for Missed Pairs
    print(f"\n[Step 4/5] Retrieving source records for {missed_count} missed pairs...")
    needed_s1_ids = {s1 for s1, _ in missed_pairs}
    needed_s2_ids = {cand for _, cand in missed_pairs if cand.startswith("S2-")}
    needed_s3_ids = {cand for _, cand in missed_pairs if cand.startswith("S3-")}

    s1_records: Dict[str, RecordInfo] = {}
    target_records: Dict[str, RecordInfo] = {}

    # Read S1
    for chunk in pd.read_csv(TRAIN_SOURCE_FILES["source1"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False):
        mask = chunk["entity_id"].isin(needed_s1_ids)
        if mask.any():
            for _, row in chunk[mask].iterrows():
                rec = extract_record_from_row(row)
                s1_records[rec.entity_id] = rec
        if len(s1_records) >= len(needed_s1_ids):
            break

    # Read S2
    if needed_s2_ids:
        for chunk in pd.read_csv(TRAIN_SOURCE_FILES["source2"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False):
            mask = chunk["entity_id"].isin(needed_s2_ids)
            if mask.any():
                for _, row in chunk[mask].iterrows():
                    rec = extract_record_from_row(row)
                    target_records[rec.entity_id] = rec
            if sum(1 for cid in needed_s2_ids if cid in target_records) >= len(needed_s2_ids):
                break
        gc.collect()

    # Read S3
    if needed_s3_ids:
        for chunk in pd.read_csv(TRAIN_SOURCE_FILES["source3"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False):
            mask = chunk["entity_id"].isin(needed_s3_ids)
            if mask.any():
                for _, row in chunk[mask].iterrows():
                    rec = extract_record_from_row(row)
                    target_records[rec.entity_id] = rec
            if sum(1 for cid in needed_s3_ids if cid in target_records) >= len(needed_s3_ids):
                break
        gc.collect()

    print(f"  Loaded {len(s1_records)} S1 records and {len(target_records)} target records.")

    # 5. Compute Diagnostic Features & Simulations
    print("\n[Step 5/5] Computing feature metrics and blocking simulations...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    miss_rows = []
    simulation_hits = {
        "exact_name": 0,
        "exact_address": 0,
        "shared_name_token_1plus": 0,
        "shared_name_token_2plus": 0,
        "shared_address_token_1plus": 0,
        "shared_address_token_2plus": 0,
        "shared_any_token_1plus": 0,
        "shared_total_token_2plus": 0,
        # Simulations with country constraint (realistic blocking)
        "country_and_exact_name": 0,
        "country_and_exact_address": 0,
        "country_and_name_token_1plus": 0,
        "country_and_name_token_2plus": 0,
        "country_and_address_token_1plus": 0,
        "country_and_address_token_2plus": 0,
        "country_and_any_token_1plus": 0,
        "country_and_total_token_2plus": 0,
    }

    counts_char = {
        "country_exact_match": 0,
        "exact_normalized_name": 0,
        "exact_normalized_address": 0,
        "different_name_script": 0,
        "missing_s1_name": 0,
        "missing_target_name": 0,
        "missing_s1_address": 0,
        "missing_target_address": 0,
        "shared_numeric_token_1plus": 0,
    }

    with open(DIAGNOSTIC_TSV_FILE, "w", encoding="utf-8") as tsv_out:
        header = [
            "s1_entity_id",
            "target_entity_id",
            "target_source",
            "s1_country",
            "target_country",
            "country_exact_match",
            "s1_name",
            "target_name",
            "exact_normalized_name",
            "s1_name_token_count",
            "target_name_token_count",
            "shared_name_token_count",
            "name_token_jaccard",
            "s1_address",
            "target_address",
            "exact_normalized_address",
            "s1_address_token_count",
            "target_address_token_count",
            "shared_address_token_count",
            "address_token_jaccard",
            "shared_numeric_token_count",
            "name_script_relation",
            "s1_name_missing",
            "target_name_missing",
            "s1_address_missing",
            "target_address_missing",
            "rule_exact_name",
            "rule_exact_address",
            "rule_name_token_1plus",
            "rule_name_token_2plus",
            "rule_address_token_1plus",
            "rule_address_token_2plus",
            "rule_any_token_1plus",
            "rule_total_token_2plus",
        ]
        tsv_out.write("\t".join(header) + "\n")

        for s1_id, cand_id in sorted(missed_pairs):
            s1 = s1_records.get(s1_id)
            target = target_records.get(cand_id)

            if not s1 or not target:
                continue

            target_source = "S2" if cand_id.startswith("S2-") else "S3"

            # Country
            c_match = bool(s1.country and target.country and s1.country == target.country)
            if c_match:
                counts_char["country_exact_match"] += 1

            # Missingness
            s1_n_miss = not bool(s1.name_norm)
            tgt_n_miss = not bool(target.name_norm)
            s1_a_miss = not bool(s1.addr_norm)
            tgt_a_miss = not bool(target.addr_norm)

            if s1_n_miss: counts_char["missing_s1_name"] += 1
            if tgt_n_miss: counts_char["missing_target_name"] += 1
            if s1_a_miss: counts_char["missing_s1_address"] += 1
            if tgt_a_miss: counts_char["missing_target_address"] += 1

            # Name features
            exact_name = bool(s1.name_norm and target.name_norm and s1.name_norm == target.name_norm)
            if exact_name:
                counts_char["exact_normalized_name"] += 1

            shared_name_tokens = s1.name_tokens & target.name_tokens
            n_shared_name = len(shared_name_tokens)
            union_name_tokens = s1.name_tokens | target.name_tokens
            name_jaccard = (n_shared_name / len(union_name_tokens)) if union_name_tokens else 0.0

            # Address features
            exact_addr = bool(s1.addr_norm and target.addr_norm and s1.addr_norm == target.addr_norm)
            if exact_addr:
                counts_char["exact_normalized_address"] += 1

            shared_addr_tokens = s1.addr_tokens & target.addr_tokens
            n_shared_addr = len(shared_addr_tokens)
            union_addr_tokens = s1.addr_tokens | target.addr_tokens
            addr_jaccard = (n_shared_addr / len(union_addr_tokens)) if union_addr_tokens else 0.0

            # Numeric tokens
            shared_num_tokens = s1.numeric_tokens & target.numeric_tokens
            n_shared_numeric = len(shared_num_tokens)
            if n_shared_numeric > 0:
                counts_char["shared_numeric_token_1plus"] += 1

            # Script comparison
            script_rel = compare_scripts(s1.name_raw, target.name_raw)
            if script_rel == "DIFFERENT":
                counts_char["different_name_script"] += 1

            # Simulation rules
            r_exact_n = exact_name
            r_exact_a = exact_addr
            r_nt_1 = n_shared_name >= 1
            r_nt_2 = n_shared_name >= 2
            r_at_1 = n_shared_addr >= 1
            r_at_2 = n_shared_addr >= 2
            r_any_1 = r_nt_1 or r_at_1
            r_tot_2 = (n_shared_name + n_shared_addr) >= 2

            if r_exact_n: simulation_hits["exact_name"] += 1
            if r_exact_a: simulation_hits["exact_address"] += 1
            if r_nt_1: simulation_hits["shared_name_token_1plus"] += 1
            if r_nt_2: simulation_hits["shared_name_token_2plus"] += 1
            if r_at_1: simulation_hits["shared_address_token_1plus"] += 1
            if r_at_2: simulation_hits["shared_address_token_2plus"] += 1
            if r_any_1: simulation_hits["shared_any_token_1plus"] += 1
            if r_tot_2: simulation_hits["shared_total_token_2plus"] += 1

            if c_match:
                if r_exact_n: simulation_hits["country_and_exact_name"] += 1
                if r_exact_a: simulation_hits["country_and_exact_address"] += 1
                if r_nt_1: simulation_hits["country_and_name_token_1plus"] += 1
                if r_nt_2: simulation_hits["country_and_name_token_2plus"] += 1
                if r_at_1: simulation_hits["country_and_address_token_1plus"] += 1
                if r_at_2: simulation_hits["country_and_address_token_2plus"] += 1
                if r_any_1: simulation_hits["country_and_any_token_1plus"] += 1
                if r_tot_2: simulation_hits["country_and_total_token_2plus"] += 1

            row_data = [
                s1.entity_id,
                target.entity_id,
                target_source,
                s1.country,
                target.country,
                "1" if c_match else "0",
                s1.name_norm,
                target.name_norm,
                "1" if exact_name else "0",
                str(len(s1.name_tokens)),
                str(len(target.name_tokens)),
                str(n_shared_name),
                f"{name_jaccard:.4f}",
                s1.addr_norm,
                target.addr_norm,
                "1" if exact_addr else "0",
                str(len(s1.addr_tokens)),
                str(len(target.addr_tokens)),
                str(n_shared_addr),
                f"{addr_jaccard:.4f}",
                str(n_shared_numeric),
                script_rel,
                "1" if s1_n_miss else "0",
                "1" if tgt_n_miss else "0",
                "1" if s1_a_miss else "0",
                "1" if tgt_a_miss else "0",
                "1" if r_exact_n else "0",
                "1" if r_exact_a else "0",
                "1" if r_nt_1 else "0",
                "1" if r_nt_2 else "0",
                "1" if r_at_1 else "0",
                "1" if r_at_2 else "0",
                "1" if r_any_1 else "0",
                "1" if r_tot_2 else "0",
            ]
            tsv_out.write("\t".join(row_data) + "\n")

            miss_rows.append({
                "s1_id": s1.entity_id,
                "target_id": target.entity_id,
                "source": target_source,
                "s1_name": s1.name_raw,
                "target_name": target.name_raw,
                "s1_addr": s1.addr_raw,
                "target_addr": target.addr_raw,
                "shared_name_tokens": list(shared_name_tokens),
                "shared_addr_tokens": list(shared_addr_tokens),
                "country_match": c_match,
                "script_rel": script_rel,
            })

    # Prepare JSON Report
    report = {
        "sampled_s1_count": len(s1_sample_set),
        "total_true_matches": total_true_matches,
        "recovered_true_matches": recovered_count,
        "missed_true_matches": missed_count,
        "candidate_recall": round(recall, 6),
        "missed_characteristics_percentages": {
            "country_exact_match": round(100.0 * counts_char["country_exact_match"] / missed_count, 2),
            "exact_normalized_name": round(100.0 * counts_char["exact_normalized_name"] / missed_count, 2),
            "exact_normalized_address": round(100.0 * counts_char["exact_normalized_address"] / missed_count, 2),
            "shared_name_token_1plus": round(100.0 * simulation_hits["shared_name_token_1plus"] / missed_count, 2),
            "shared_name_token_2plus": round(100.0 * simulation_hits["shared_name_token_2plus"] / missed_count, 2),
            "shared_address_token_1plus": round(100.0 * simulation_hits["shared_address_token_1plus"] / missed_count, 2),
            "shared_address_token_2plus": round(100.0 * simulation_hits["shared_address_token_2plus"] / missed_count, 2),
            "shared_any_token_1plus": round(100.0 * simulation_hits["shared_any_token_1plus"] / missed_count, 2),
            "shared_total_token_2plus": round(100.0 * simulation_hits["shared_total_token_2plus"] / missed_count, 2),
            "different_name_script": round(100.0 * counts_char["different_name_script"] / missed_count, 2),
            "missing_name": round(100.0 * (counts_char["missing_s1_name"] + counts_char["missing_target_name"]) / (2 * missed_count), 2),
            "missing_address": round(100.0 * (counts_char["missing_s1_address"] + counts_char["missing_target_address"]) / (2 * missed_count), 2),
            "shared_numeric_token_1plus": round(100.0 * counts_char["shared_numeric_token_1plus"] / missed_count, 2),
        },
        "blocking_simulation_country_constrained": {
            "country_and_exact_name": {
                "missed_recovered": simulation_hits["country_and_exact_name"],
                "estimated_recall": round((recovered_count + simulation_hits["country_and_exact_name"]) / total_true_matches, 6),
            },
            "country_and_exact_address": {
                "missed_recovered": simulation_hits["country_and_exact_address"],
                "estimated_recall": round((recovered_count + simulation_hits["country_and_exact_address"]) / total_true_matches, 6),
            },
            "country_and_name_token_1plus": {
                "missed_recovered": simulation_hits["country_and_name_token_1plus"],
                "estimated_recall": round((recovered_count + simulation_hits["country_and_name_token_1plus"]) / total_true_matches, 6),
            },
            "country_and_name_token_2plus": {
                "missed_recovered": simulation_hits["country_and_name_token_2plus"],
                "estimated_recall": round((recovered_count + simulation_hits["country_and_name_token_2plus"]) / total_true_matches, 6),
            },
            "country_and_address_token_1plus": {
                "missed_recovered": simulation_hits["country_and_address_token_1plus"],
                "estimated_recall": round((recovered_count + simulation_hits["country_and_address_token_1plus"]) / total_true_matches, 6),
            },
            "country_and_address_token_2plus": {
                "missed_recovered": simulation_hits["country_and_address_token_2plus"],
                "estimated_recall": round((recovered_count + simulation_hits["country_and_address_token_2plus"]) / total_true_matches, 6),
            },
            "country_and_any_token_1plus": {
                "missed_recovered": simulation_hits["country_and_any_token_1plus"],
                "estimated_recall": round((recovered_count + simulation_hits["country_and_any_token_1plus"]) / total_true_matches, 6),
            },
            "country_and_total_token_2plus": {
                "missed_recovered": simulation_hits["country_and_total_token_2plus"],
                "estimated_recall": round((recovered_count + simulation_hits["country_and_total_token_2plus"]) / total_true_matches, 6),
            },
        },
    }

    with open(DIAGNOSTIC_REPORT_FILE, "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2)

    # Human-Readable Summary
    print("\n" + "=" * 70)
    print("CANDIDATE MISS DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Sampled S1 entities : {len(s1_sample_set)}")
    print(f"True matches        : {total_true_matches}")
    print(f"Recovered           : {recovered_count} ({recall * 100:.2f}%)")
    print(f"Missed              : {missed_count} ({(1 - recall) * 100:.2f}%)")

    print("\nMissed-pair characteristics:")
    for metric, pct in report["missed_characteristics_percentages"].items():
        print(f"  {metric:30s}: {pct:6.2f}%")

    print("\nBlocking simulation (Country constrained):")
    print(f"  {'Rule':32s} {'Missed Recovered':18s} {'Estimated Recall':16s}")
    print("  " + "-" * 68)
    for rule, data in report["blocking_simulation_country_constrained"].items():
        m_rec = data["missed_recovered"]
        est_r = data["estimated_recall"]
        print(f"  {rule:32s} {m_rec:<18d} {est_r * 100:6.2f}%")

    print("\n" + "=" * 70)
    print("TOP 20 MISSED EXAMPLES")
    print("=" * 70)
    for i, ex in enumerate(miss_rows[:20], 1):
        print(f"\n[{i:02d}] {ex['s1_id']}  <--->  {ex['target_id']} ({ex['source']}) | Country Match: {ex['country_match']} | Script: {ex['script_rel']}")
        print(f"     S1 Name     : {ex['s1_name']}")
        print(f"     Target Name : {ex['target_name']}")
        print(f"     S1 Address  : {ex['s1_addr']}")
        print(f"     Target Addr : {ex['target_addr']}")
        print(f"     Shared Name Tokens : {ex['shared_name_tokens']}")
        print(f"     Shared Addr Tokens : {ex['shared_addr_tokens']}")

    print("\n" + "=" * 70)
    print(f"Detailed TSV written to : {DIAGNOSTIC_TSV_FILE}")
    print(f"JSON Report written to  : {DIAGNOSTIC_REPORT_FILE}")
    print(f"Completed in {time.time() - t0:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostic()