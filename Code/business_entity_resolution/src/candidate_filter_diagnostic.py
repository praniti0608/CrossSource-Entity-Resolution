"""
candidate_filter_diagnostic.py

Diagnostic tool to analyze missed true matches from candidate_generation_fast.py.
Amazon ML Challenge 2026 - Business Entity Resolution.

Algorithm:
1. Reproduces exact 1000 S1 sample.
2. Identifies 1101 missed true matches.
3. Computes S1 and Target token frequencies via memory-bounded streaming.
4. Simulates token filtering and stratified voting per missed pair (Fixed to per-field max).
5. Outputs detailed failure reasons and aggregate diagnostic metrics.
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


# --- Configuration ---
CHUNKSIZE = 200_000
COMMON_TOKEN_THRESHOLD = 50
TARGET_TOKEN_MAX_FREQ = 1000
SEED = 42

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

DIAGNOSTIC_TSV_FILE = OUTPUT_DIR / "candidate_filter_diagnostic.tsv"
DIAGNOSTIC_REPORT_FILE = OUTPUT_DIR / "candidate_filter_diagnostic_report.json"


# --- Normalization & Tokenization Utilities ---
def _default_normalize(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return ""
    text_str = unicodedata.normalize("NFKC", str(text))
    text_str = re.sub(r"\s+", " ", text_str).strip().casefold()
    return text_str

def normalize_name(text: Optional[str]) -> str:
    if text is None or pd.isna(text): return ""
    if prep_mod is not None:
        for fn_name in ("normalize_business_name", "normalize_name", "normalize_text", "clean_text"):
            if hasattr(prep_mod, fn_name):
                res = getattr(prep_mod, fn_name)(text)
                return "" if res is None or pd.isna(res) else str(res).strip()
    return _default_normalize(text)

def normalize_address(text: Optional[str]) -> str:
    if text is None or pd.isna(text): return ""
    if prep_mod is not None:
        for fn_name in ("normalize_business_address", "normalize_address", "normalize_text", "clean_text"):
            if hasattr(prep_mod, fn_name):
                res = getattr(prep_mod, fn_name)(text)
                return "" if res is None or pd.isna(res) else str(res).strip()
    return _default_normalize(text)

def normalize_country(text: Optional[str]) -> str:
    if text is None or pd.isna(text): return ""
    if prep_mod is not None:
        for fn_name in ("normalize_country", "clean_country", "normalize_text"):
            if hasattr(prep_mod, fn_name):
                res = getattr(prep_mod, fn_name)(text)
                return "" if res is None or pd.isna(res) else str(res).strip()
    return _default_normalize(text)

def safe_parse_tokens(raw_val: any, norm_val: str) -> Set[str]:
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

def extract_normalized_and_tokens(row: pd.Series) -> Tuple[str, str, str, str, Set[str], Set[str]]:
    eid = str(row.get("entity_id", "")).strip()
    
    country = row.get("country_normalized")
    if pd.isna(country) or not country: country = normalize_country(row.get("country"))
    
    name = row.get("business_name_normalized")
    if pd.isna(name) or not name: name = normalize_name(row.get("business_name"))
        
    addr = row.get("business_address_normalized")
    if pd.isna(addr) or not addr: addr = normalize_address(row.get("business_address"))
        
    c, n, a = str(country).strip(), str(name).strip(), str(addr).strip()
    n_toks = safe_parse_tokens(row.get("business_name_tokens", None), n)
    a_toks = safe_parse_tokens(row.get("business_address_tokens", None), a)
        
    return eid, c, n, a, n_toks, a_toks


class RecordInfo:
    __slots__ = ("entity_id", "country", "name_tokens", "addr_tokens")
    def __init__(self, entity_id: str, country: str, name_tokens: Set[str], addr_tokens: Set[str]):
        self.entity_id = entity_id
        self.country = country
        self.name_tokens = name_tokens
        self.addr_tokens = addr_tokens


def get_1000_s1_sample(seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    reservoir = []
    n_seen = 0
    reader = pd.read_csv(TRAIN_SOURCE_FILES["source1"], sep="\t", usecols=["entity_id"], chunksize=CHUNKSIZE, dtype=str)
    for chunk in reader:
        for eid in chunk["entity_id"]:
            if pd.isna(eid) or not str(eid).strip(): continue
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
    # Enforce safe terminal output encoding
    if hasattr(sys.stdout, 'reconfigure'):
        try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except: pass
    if hasattr(sys.stderr, 'reconfigure'):
        try: sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except: pass

    print("=======================================================")
    print("CANDIDATE FILTER DIAGNOSTIC (Amazon ML Challenge 2026)")
    print("=======================================================")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # --- STEP 1 & 2: Load Sample and Ground Truth ---
    print("\n[Step 1 & 2] Sampling 1000 S1 records & reading true pairs...")
    s1_ids = get_1000_s1_sample(SEED)
    sampled_set = set(s1_ids)

    true_pairs = set()
    gt_reader = pd.read_csv(GROUND_TRUTH_FILE, sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
    for chunk in gt_reader:
        mask = chunk["source1_entity_id"].isin(sampled_set)
        if mask.any():
            for _, row in chunk[mask].iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                raw_matched = str(row.get("matched_entity_ids", "")).strip()
                if raw_matched:
                    for cand_id in raw_matched.split(","):
                        if cand_id.strip():
                            true_pairs.add((s1_id, cand_id.strip()))

    total_true_matches = len(true_pairs)
    
    # --- STEP 3: Load Existing Candidates & Identify Misses ---
    print(f"\n[Step 3] Parsing {OUTPUT_PAIRS_FILE.name} for missed pairs...")
    if not OUTPUT_PAIRS_FILE.exists():
        raise FileNotFoundError(f"Missing {OUTPUT_PAIRS_FILE}. Run fast baseline first.")
        
    candidate_pairs = set()
    with open(OUTPUT_PAIRS_FILE, "r", encoding="utf-8") as f:
        next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2 and parts[0] in sampled_set:
                candidate_pairs.add((parts[0], parts[1]))
                
    recovered_pairs = true_pairs & candidate_pairs
    missed_pairs = true_pairs - candidate_pairs
    missed_count = len(missed_pairs)

    print(f"  Sampled S1 entities: {len(sampled_set)}")
    print(f"  Total true matches:  {total_true_matches}")
    print(f"  Recovered matches:   {len(recovered_pairs)}")
    print(f"  Missed matches:      {missed_count}")
    
    if len(sampled_set) != 1000 or total_true_matches != 3458 or missed_count != 1101:
        print("\n  >> WARNING: Counts do not perfectly match expected previous state.")
        print("  >> The diagnostic will still proceed accurately for the currently detected pairs.")

    if missed_count == 0:
        print("No missed pairs to diagnose! Exiting.")
        return

    # --- STEP 4: Retrieve Required S1 & Target Records ---
    print("\n[Step 4] Retrieving required records & building S1 Token Filters...")
    needed_s1_ids = {s1 for s1, _ in missed_pairs}
    needed_s2_ids = {cand for _, cand in missed_pairs if cand.startswith("S2-")}
    needed_s3_ids = {cand for _, cand in missed_pairs if cand.startswith("S3-")}

    s1_records: Dict[str, RecordInfo] = {}
    target_records: Dict[str, RecordInfo] = {}
    
    s1_nt_freq = defaultdict(int)
    s1_at_freq = defaultdict(int)

    reader = pd.read_csv(TRAIN_SOURCE_FILES["source1"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
    for chunk in reader:
        mask = chunk["entity_id"].isin(sampled_set)
        if mask.any():
            for _, row in chunk[mask].iterrows():
                eid, c, _, _, n_toks, a_toks = extract_normalized_and_tokens(row)
                if c:
                    for tk in n_toks: s1_nt_freq[(c, tk)] += 1
                    for tk in a_toks: s1_at_freq[(c, tk)] += 1
                if eid in needed_s1_ids:
                    s1_records[eid] = RecordInfo(eid, c, n_toks, a_toks)

    s1_nt_eligible = {k for k, count in s1_nt_freq.items() if count <= COMMON_TOKEN_THRESHOLD}
    s1_at_eligible = {k for k, count in s1_at_freq.items() if count <= COMMON_TOKEN_THRESHOLD}

    for source_file, needed_ids in [(TRAIN_SOURCE_FILES["source2"], needed_s2_ids), 
                                    (TRAIN_SOURCE_FILES["source3"], needed_s3_ids)]:
        if needed_ids:
            for chunk in pd.read_csv(source_file, sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False):
                mask = chunk["entity_id"].isin(needed_ids)
                if mask.any():
                    for _, row in chunk[mask].iterrows():
                        eid, c, _, _, n_toks, a_toks = extract_normalized_and_tokens(row)
                        target_records[eid] = RecordInfo(eid, c, n_toks, a_toks)
                if sum(1 for cid in needed_ids if cid in target_records) >= len(needed_ids):
                    break
            gc.collect()

    # --- STEP 5: Compute Target Frequencies for Required Tokens ---
    print("\n[Step 5] Computing Target Token Frequencies via streaming...")
    s2_nt_freq = defaultdict(int)
    s2_at_freq = defaultdict(int)
    s3_nt_freq = defaultdict(int)
    s3_at_freq = defaultdict(int)
    
    for label, target_path, nt_freq, at_freq in [
        ("S2", TRAIN_SOURCE_FILES["source2"], s2_nt_freq, s2_at_freq),
        ("S3", TRAIN_SOURCE_FILES["source3"], s3_nt_freq, s3_at_freq)
    ]:
        reader = pd.read_csv(target_path, sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
        for chunk in reader:
            for row in chunk.itertuples(index=False):
                c = normalize_country(getattr(row, "country", ""))
                if not c: continue
                
                n = normalize_name(getattr(row, "business_name", ""))
                a = normalize_address(getattr(row, "business_address", ""))
                
                for tk in safe_parse_tokens(getattr(row, "business_name_tokens", None), n):
                    if (c, tk) in s1_nt_eligible: nt_freq[(c, tk)] += 1
                for tk in safe_parse_tokens(getattr(row, "business_address_tokens", None), a):
                    if (c, tk) in s1_at_eligible: at_freq[(c, tk)] += 1
        gc.collect()

    # --- STEP 6: Simulate Diagnostic Rules ---
    print("\n[Step 6] Diagnosing failure reasons for missed pairs...")
    
    report_data = {
        "total_missed_pairs": missed_count,
        "pairs_with_1plus_eligible_name_token": 0,
        "pairs_with_1plus_eligible_address_token": 0,
        "pairs_with_1plus_eligible_strong_token": 0,
        "pairs_with_2plus_eligible_normal_tokens": 0,
        "pairs_passing_current_rule": 0,
        "pairs_rejected_by_current_rule": 0,
    }
    
    failure_reasons = {
        "S1_TOKEN_FILTER": 0,
        "TARGET_TOKEN_FILTER": 0,
        "TOKEN_VOTE_THRESHOLD": 0,
        "NAME_ONLY_SIGNAL": 0,
        "ADDRESS_ONLY_SIGNAL": 0,
        "BOTH_NAME_AND_ADDRESS_FILTERED": 0,
        "NO_ELIGIBLE_TOKEN": 0,
        "OTHER": 0
    }
    
    source_split = {"S2": 0, "S3": 0}

    with open(DIAGNOSTIC_TSV_FILE, "w", encoding="utf-8") as tsv_out:
        header = [
            "source1_entity_id", "target_entity_id", "target_source", "country_match",
            "shared_name_tokens", "shared_address_tokens", 
            "eligible_name_tokens", "eligible_address_tokens",
            "eligible_strong_tokens", "eligible_normal_tokens",
            "passes_current_rule", "failure_reason"
        ]
        tsv_out.write("\t".join(header) + "\n")

        for s1_id, cand_id in sorted(missed_pairs):
            s1 = s1_records.get(s1_id)
            target = target_records.get(cand_id)
            
            if not s1 or not target: continue
            target_source = "S2" if cand_id.startswith("S2-") else "S3"
            source_split[target_source] += 1
            
            country_match = bool(s1.country and target.country and s1.country == target.country)
            
            shared_nt = s1.name_tokens & target.name_tokens
            shared_at = s1.addr_tokens & target.addr_tokens
            
            s1_survived_nt = {tk for tk in shared_nt if (s1.country, tk) in s1_nt_eligible}
            s1_survived_at = {tk for tk in shared_at if (s1.country, tk) in s1_at_eligible}
            
            target_nt_dict = s2_nt_freq if target_source == "S2" else s3_nt_freq
            target_at_dict = s2_at_freq if target_source == "S2" else s3_at_freq
            
            eligible_nt = {tk for tk in s1_survived_nt if target_nt_dict.get((s1.country, tk), 0) <= TARGET_TOKEN_MAX_FREQ}
            eligible_at = {tk for tk in s1_survived_at if target_at_dict.get((s1.country, tk), 0) <= TARGET_TOKEN_MAX_FREQ}
            
            # Count name votes
            n_strong = sum(1 for tk in eligible_nt if target_nt_dict.get((s1.country, tk), 0) <= 20)
            n_normal = len(eligible_nt) - n_strong
            
            # Count address votes
            a_strong = sum(1 for tk in eligible_at if target_at_dict.get((s1.country, tk), 0) <= 20)
            a_normal = len(eligible_at) - a_strong
            
            passes_n = (n_strong >= 1) or (n_normal >= 2)
            passes_a = (a_strong >= 1) or (a_normal >= 2)
            passes_rule = country_match and (passes_n or passes_a)
            
            # The current generator evaluates name and address separately. 
            # We track the MAXIMUM across fields so it perfectly reflects the rule's threshold.
            eligible_strong_count = max(n_strong, a_strong)
            eligible_normal_count = max(n_normal, a_normal)

            if country_match and (eligible_strong_count >= 1 or eligible_normal_count >= 2):
                assert passes_rule is True, f"Rule consistency assertion failed for {s1_id} -> {cand_id}"
            
            if passes_rule:
                failure_reason = "OTHER"
            elif not country_match:
                failure_reason = "OTHER" # (Could be Country Mismatch, aggregating into OTHER)
            elif not shared_nt and not shared_at:
                failure_reason = "NO_ELIGIBLE_TOKEN"
            elif not s1_survived_nt and not s1_survived_at:
                failure_reason = "S1_TOKEN_FILTER"
            elif not eligible_nt and not eligible_at:
                failure_reason = "TARGET_TOKEN_FILTER"
            elif eligible_nt and not eligible_at:
                failure_reason = "NAME_ONLY_SIGNAL"
            elif eligible_at and not eligible_nt:
                failure_reason = "ADDRESS_ONLY_SIGNAL"
            else:
                failure_reason = "TOKEN_VOTE_THRESHOLD"
                
            failure_reasons[failure_reason] += 1
            
            if eligible_nt: report_data["pairs_with_1plus_eligible_name_token"] += 1
            if eligible_at: report_data["pairs_with_1plus_eligible_address_token"] += 1
            if eligible_strong_count >= 1: report_data["pairs_with_1plus_eligible_strong_token"] += 1
            if eligible_normal_count >= 2: report_data["pairs_with_2plus_eligible_normal_tokens"] += 1
            if passes_rule: report_data["pairs_passing_current_rule"] += 1
            else: report_data["pairs_rejected_by_current_rule"] += 1
            
            row = [
                s1.entity_id, target.entity_id, target_source,
                str(country_match),
                ",".join(shared_nt), ",".join(shared_at),
                ",".join(eligible_nt), ",".join(eligible_at),
                str(eligible_strong_count), str(eligible_normal_count),
                str(passes_rule), failure_reason
            ]
            tsv_out.write("\t".join(row) + "\n")

    report_data["failure_reason_distribution"] = failure_reasons
    report_data["target_source_distribution"] = source_split

    with open(DIAGNOSTIC_REPORT_FILE, "w", encoding="utf-8") as jf:
        json.dump(report_data, jf, indent=2)

    # --- STEP 7: Terminal Output ---
    print("\n" + "=" * 70)
    print("CANDIDATE FILTER DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"Total missed pairs                      : {missed_count}")
    print(f"Pairs with >=1 eligible name token      : {report_data['pairs_with_1plus_eligible_name_token']}")
    print(f"Pairs with >=1 eligible address token   : {report_data['pairs_with_1plus_eligible_address_token']}")
    print(f"Eligible strong >=1                     : {report_data['pairs_with_1plus_eligible_strong_token']}")
    print(f"Eligible normal >=2                     : {report_data['pairs_with_2plus_eligible_normal_tokens']}")
    print(f"Passes current rule                     : {report_data['pairs_passing_current_rule']}")
    print(f"Pairs rejected by current rule          : {report_data['pairs_rejected_by_current_rule']}")
    
    print("\nFailure-reason distribution:")
    for reason, count in failure_reasons.items():
        print(f"  {reason:30s} : {count}")
        
    print("\nTarget Source distribution of missed pairs:")
    for src, count in source_split.items():
        print(f"  {src:30s} : {count}")

    print("\n" + "=" * 70)
    print(f"Diagnostic TSV saved to   : {DIAGNOSTIC_TSV_FILE}")
    print(f"Diagnostic JSON saved to  : {DIAGNOSTIC_REPORT_FILE}")
    print(f"Completed in {time.time() - t0:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    run_diagnostic()