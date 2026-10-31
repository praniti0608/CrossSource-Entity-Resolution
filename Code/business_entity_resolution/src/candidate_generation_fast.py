from __future__ import annotations

import os
import sys
import gc
import json
import time
import ast
import argparse
import unicodedata
import re
import heapq
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple
from collections import defaultdict

import pandas as pd

try:
    import psutil
except ImportError:
    psutil = None

# Dynamic import of preprocessing logic
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
TARGET_TOKEN_MAX_FREQ = 2500
MAX_CANDIDATES_PER_TARGET = 500
S1_BLOCK_SIZE = 250000


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
OUTPUT_PAIRS_FILE = OUTPUT_DIR / "candidate_pairs_fast_block_01.tsv"


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


def extract_numeric_tokens(text: str) -> Set[str]:
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


def extract_normalized_and_tokens(row: pd.Series) -> Tuple[str, str, str, str, Set[str], Set[str], Set[str]]:
    eid = str(row.get("entity_id", "")).strip()
    
    country = row.get("country_normalized")
    if pd.isna(country) or not country:
        country = normalize_country(row.get("country"))
    
    name_raw = str(row.get("business_name", ""))
    name = row.get("business_name_normalized")
    if pd.isna(name) or not name:
        name = normalize_name(name_raw)
        
    addr_raw = str(row.get("business_address", ""))
    addr = row.get("business_address_normalized")
    if pd.isna(addr) or not addr:
        addr = normalize_address(addr_raw)
        
    c, n, a = str(country).strip(), str(name).strip(), str(addr).strip()
    
    n_toks = safe_parse_tokens(row.get("business_name_tokens", None), n)
    a_toks = safe_parse_tokens(row.get("business_address_tokens", None), a)
    num_toks = extract_numeric_tokens(addr_raw if addr_raw.strip() else a)
        
    return eid, c, n, a, n_toks, a_toks, num_toks


class _CandItem:
    """Helper class to maintain a min-heap of the 'best' candidates within the bounded cap."""
    __slots__ = ('score', 'at_c', 'nt_c', 's1_id')

    def __init__(self, score: int, at_c: int, nt_c: int, s1_id: str):
        self.score = score
        self.at_c = at_c
        self.nt_c = nt_c
        self.s1_id = s1_id

    def __lt__(self, other: '_CandItem') -> bool:
        if self.score != other.score:
            return self.score < other.score
        if self.at_c != other.at_c:
            return self.at_c < other.at_c
        if self.nt_c != other.nt_c:
            return self.nt_c < other.nt_c
        return self.s1_id > other.s1_id


# --- Core Pipeline ---
def get_memory_usage_mb() -> float:
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


def precompute_global_tokens() -> Tuple[Set[Tuple[str, str]], Set[Tuple[str, str]], Set[Tuple[str, str]], Set[Tuple[str, str]]]:
    """
    Globally precomputes token frequencies across ALL Source-1, Source-2, and Source-3 records,
    maintaining separate target token frequency thresholds for S2 and S3.
    """
    print("\n[Global Token Vocabulary Pass 1] Streaming S1 to calculate global token frequencies...")
    s1_nt_freq = defaultdict(int)
    s1_at_freq = defaultdict(int)
    
    s1_reader = pd.read_csv(TRAIN_SOURCE_FILES["source1"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
    for chunk in s1_reader:
        for row in chunk.itertuples(index=False):
            country = getattr(row, "country_normalized", "")
            if not country: country = normalize_country(getattr(row, "country", ""))
            c = str(country).strip()
            if not c: continue
            
            name = getattr(row, "business_name_normalized", "")
            if not name: name = normalize_name(getattr(row, "business_name", ""))
            addr = getattr(row, "business_address_normalized", "")
            if not addr: addr = normalize_address(getattr(row, "business_address", ""))
            
            n_toks = safe_parse_tokens(getattr(row, "business_name_tokens", None), name)
            a_toks = safe_parse_tokens(getattr(row, "business_address_tokens", None), addr)
            
            for tk in n_toks: s1_nt_freq[(c, tk)] += 1
            for tk in a_toks: s1_at_freq[(c, tk)] += 1

    total_s1_name_tokens = len(s1_nt_freq)
    total_s1_addr_tokens = len(s1_at_freq)
    
    eligible_s1_nt = {k for k, v in s1_nt_freq.items() if v <= COMMON_TOKEN_THRESHOLD}
    eligible_s1_at = {k for k, v in s1_at_freq.items() if v <= COMMON_TOKEN_THRESHOLD}
    
    del s1_nt_freq
    del s1_at_freq
    gc.collect()

    print(f"  Total unique S1 name tokens: {total_s1_name_tokens:,}")
    print(f"  Eligible S1 name tokens: {len(eligible_s1_nt):,}")
    print(f"  Total unique S1 address tokens: {total_s1_addr_tokens:,}")
    print(f"  Eligible S1 address tokens: {len(eligible_s1_at):,}")

    print("\n[Global Token Vocabulary Pass 2] Streaming S2 & S3 separately for target token frequencies...")
    
    s2_nt_freq = defaultdict(int)
    s2_at_freq = defaultdict(int)
    s3_nt_freq = defaultdict(int)
    s3_at_freq = defaultdict(int)

    # Stream S2
    s2_reader = pd.read_csv(TRAIN_SOURCE_FILES["source2"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
    for chunk in s2_reader:
        for row in chunk.itertuples(index=False):
            country = getattr(row, "country_normalized", "")
            if not country: country = normalize_country(getattr(row, "country", ""))
            c = str(country).strip()
            if not c: continue
            
            name = getattr(row, "business_name_normalized", "")
            if not name: name = normalize_name(getattr(row, "business_name", ""))
            addr = getattr(row, "business_address_normalized", "")
            if not addr: addr = normalize_address(getattr(row, "business_address", ""))
            
            n_toks = safe_parse_tokens(getattr(row, "business_name_tokens", None), name)
            a_toks = safe_parse_tokens(getattr(row, "business_address_tokens", None), addr)
            
            for tk in n_toks:
                if (c, tk) in eligible_s1_nt:
                    s2_nt_freq[(c, tk)] += 1
            for tk in a_toks:
                if (c, tk) in eligible_s1_at:
                    s2_at_freq[(c, tk)] += 1

    # Stream S3
    s3_reader = pd.read_csv(TRAIN_SOURCE_FILES["source3"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
    for chunk in s3_reader:
        for row in chunk.itertuples(index=False):
            country = getattr(row, "country_normalized", "")
            if not country: country = normalize_country(getattr(row, "country", ""))
            c = str(country).strip()
            if not c: continue
            
            name = getattr(row, "business_name_normalized", "")
            if not name: name = normalize_name(getattr(row, "business_name", ""))
            addr = getattr(row, "business_address_normalized", "")
            if not addr: addr = normalize_address(getattr(row, "business_address", ""))
            
            n_toks = safe_parse_tokens(getattr(row, "business_name_tokens", None), name)
            a_toks = safe_parse_tokens(getattr(row, "business_address_tokens", None), addr)
            
            for tk in n_toks:
                if (c, tk) in eligible_s1_nt:
                    s3_nt_freq[(c, tk)] += 1
            for tk in a_toks:
                if (c, tk) in eligible_s1_at:
                    s3_at_freq[(c, tk)] += 1

    s2_valid_nt = {k for k in eligible_s1_nt if s2_nt_freq[k] <= TARGET_TOKEN_MAX_FREQ}
    s2_valid_at = {k for k in eligible_s1_at if s2_at_freq[k] <= TARGET_TOKEN_MAX_FREQ}
    s3_valid_nt = {k for k in eligible_s1_nt if s3_nt_freq[k] <= TARGET_TOKEN_MAX_FREQ}
    s3_valid_at = {k for k in eligible_s1_at if s3_at_freq[k] <= TARGET_TOKEN_MAX_FREQ}

    del eligible_s1_nt
    del eligible_s1_at
    del s2_nt_freq
    del s2_at_freq
    del s3_nt_freq
    del s3_at_freq
    gc.collect()

    print(f"  S2 Target-eligible name vocabulary: {len(s2_valid_nt):,}")
    print(f"  S2 Target-eligible address vocabulary: {len(s2_valid_at):,}")
    print(f"  S3 Target-eligible name vocabulary: {len(s3_valid_nt):,}")
    print(f"  S3 Target-eligible address vocabulary: {len(s3_valid_at):,}")

    return s2_valid_nt, s2_valid_at, s3_valid_nt, s3_valid_at


def run_pilot_block():
    print("\n=======================================================")
    print("FAST BASELINE: S1 PILOT BLOCK RUN (250K ENTITIES)")
    print("=======================================================")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_global_start = time.time()
    
    s2_valid_nt, s2_valid_at, s3_valid_nt, s3_valid_at = precompute_global_tokens()

    print(f"\n1. Loading exactly {S1_BLOCK_SIZE:,} S1 entities for Block 01...")
    
    s1_ids = set()
    s1_name_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    s1_addr_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    s1_num_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    
    s1_reader = pd.read_csv(TRAIN_SOURCE_FILES["source1"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
    loaded_count = 0
    for chunk in s1_reader:
        if loaded_count >= S1_BLOCK_SIZE:
            break
            
        if loaded_count + len(chunk) > S1_BLOCK_SIZE:
            chunk = chunk.head(S1_BLOCK_SIZE - loaded_count)
            
        for _, row in chunk.iterrows():
            eid, c, n, a, n_toks, a_toks, num_toks = extract_normalized_and_tokens(row)
            if not eid: continue
            
            s1_ids.add(eid)
            
            if c and n: s1_name_map[(c, n)].add(eid)
            if c and a: s1_addr_map[(c, a)].add(eid)
            if c:
                for tk in num_toks:
                    s1_num_map[(c, tk)].add(eid)
                    
        loaded_count += len(chunk)

    print(f"   Indexed {len(s1_ids)} S1 entities for Block 01.")
    
    with open(OUTPUT_PAIRS_FILE, "w", encoding="utf-8") as out_fh:
        out_fh.write("source1_entity_id\tcandidate_entity_id\tcandidate_source\tmethods\n")

    runtimes = {}
    total_pairs_written = 0
    
    for source_label, target_path in [("S2", TRAIN_SOURCE_FILES["source2"]), ("S3", TRAIN_SOURCE_FILES["source3"])]:
        print(f"\n--- Processing {source_label} for Block 01 ---")
        t_src_start = time.time()
        
        # Select source-specific valid token sets
        current_valid_nt = s2_valid_nt if source_label == "S2" else s3_valid_nt
        current_valid_at = s2_valid_at if source_label == "S2" else s3_valid_at

        # Build source-specific token maps for the current S1 block
        s1_nt_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        s1_at_map: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        
        s1_reader = pd.read_csv(TRAIN_SOURCE_FILES["source1"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
        loaded_count = 0
        for chunk in s1_reader:
            if loaded_count >= S1_BLOCK_SIZE:
                break
            if loaded_count + len(chunk) > S1_BLOCK_SIZE:
                chunk = chunk.head(S1_BLOCK_SIZE - loaded_count)
                
            for _, row in chunk.iterrows():
                eid, c, n, a, n_toks, a_toks, _ = extract_normalized_and_tokens(row)
                if not eid: continue
                if c:
                    for tk in n_toks:
                        if (c, tk) in current_valid_nt:
                            s1_nt_map[(c, tk)].add(eid)
                    for tk in a_toks:
                        if (c, tk) in current_valid_at:
                            s1_at_map[(c, tk)].add(eid)
            loaded_count += len(chunk)

        total_rows_scanned = 0
        target_reader = pd.read_csv(target_path, sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
        with open(OUTPUT_PAIRS_FILE, "a", encoding="utf-8") as out_fh:
            for chunk in target_reader:
                chunk_matches = defaultdict(set)
                
                for row in chunk.itertuples(index=False):
                    eid_raw = getattr(row, "entity_id", "")
                    if pd.isna(eid_raw) or not str(eid_raw).strip(): continue
                    
                    country = getattr(row, "country_normalized", "")
                    if not country: country = normalize_country(getattr(row, "country", ""))
                    name = getattr(row, "business_name_normalized", "")
                    if not name: name = normalize_name(getattr(row, "business_name", ""))
                    addr = getattr(row, "business_address_normalized", "")
                    if not addr: addr = normalize_address(getattr(row, "business_address", ""))
                    
                    cand_id = str(eid_raw).strip()
                    c, n, a = str(country).strip(), str(name).strip(), str(addr).strip()
                    
                    if c:
                        row_exact_n = set()
                        row_exact_a = set()
                        row_nt_count = defaultdict(int)
                        row_at_count = defaultdict(int)
                        row_num_count = defaultdict(int)
                        row_methods = defaultdict(set)

                        if n and (c, n) in s1_name_map:
                            for s1_id in s1_name_map[(c, n)]:
                                row_exact_n.add(s1_id)
                                row_methods[s1_id].add("exact_name")
                                
                        if a and (c, a) in s1_addr_map:
                            for s1_id in s1_addr_map[(c, a)]:
                                row_exact_a.add(s1_id)
                                row_methods[s1_id].add("exact_address")
                                
                        n_toks = safe_parse_tokens(getattr(row, "business_name_tokens", None), n)
                        for tk in n_toks:
                            if (c, tk) in s1_nt_map:
                                for s1_id in s1_nt_map[(c, tk)]:
                                    row_nt_count[s1_id] += 1
                                    row_methods[s1_id].add("name_token")
                                    
                        a_toks = safe_parse_tokens(getattr(row, "business_address_tokens", None), a)
                        for tk in a_toks:
                            if (c, tk) in s1_at_map:
                                for s1_id in s1_at_map[(c, tk)]:
                                    row_at_count[s1_id] += 1
                                    row_methods[s1_id].add("address_token")
                                    
                        num_toks = extract_numeric_tokens(str(getattr(row, "business_address", "")) or a)
                        for tk in num_toks:
                            if (c, tk) in s1_num_map:
                                for s1_id in s1_num_map[(c, tk)]:
                                    row_num_count[s1_id] += 1

                        all_matched_s1 = set(row_methods.keys())
                        
                        if len(all_matched_s1) <= MAX_CANDIDATES_PER_TARGET:
                            for s1_id in all_matched_s1:
                                chunk_matches[(s1_id, cand_id)].update(row_methods[s1_id])
                        else:
                            exact_s1s = row_exact_n | row_exact_a
                            non_exact_s1s = all_matched_s1 - exact_s1s
                            
                            for s1_id in exact_s1s:
                                chunk_matches[(s1_id, cand_id)].update(row_methods[s1_id])
                                
                            remaining_slots = MAX_CANDIDATES_PER_TARGET - len(exact_s1s)
                            if remaining_slots > 0 and non_exact_s1s:
                                candidate_gen = (
                                    (
                                        (3 * row_at_count[s1_id]) + (2 * row_nt_count[s1_id]) + row_num_count[s1_id],
                                        row_at_count[s1_id],
                                        row_nt_count[s1_id],
                                        s1_id
                                    )
                                    for s1_id in non_exact_s1s
                                )
                                
                                top_candidates = []
                                for item_tuple in candidate_gen:
                                    item = _CandItem(item_tuple[0], item_tuple[1], item_tuple[2], item_tuple[3])
                                    if len(top_candidates) < remaining_slots:
                                        heapq.heappush(top_candidates, item)
                                    else:
                                        heapq.heappushpop(top_candidates, item)
                                
                                for item in top_candidates:
                                    s1_id = item.s1_id
                                    chunk_matches[(s1_id, cand_id)].update(row_methods[s1_id])
                    
                    if len(chunk_matches) >= 100_000:
                        lines = []
                        for (s1_id, c_id), methods in chunk_matches.items():
                            methods_str = "|".join(sorted(methods))
                            lines.append(f"{s1_id}\t{c_id}\t{source_label}\t{methods_str}\n")
                            total_pairs_written += 1
                            
                        out_fh.writelines(lines)
                        chunk_matches.clear()

                if chunk_matches:
                    lines = []
                    for (s1_id, c_id), methods in chunk_matches.items():
                        methods_str = "|".join(sorted(methods))
                        lines.append(f"{s1_id}\t{c_id}\t{source_label}\t{methods_str}\n")
                        total_pairs_written += 1
                        
                    out_fh.writelines(lines)
                    chunk_matches.clear()
                    
                total_rows_scanned += len(chunk)
                if total_rows_scanned % 1_000_000 == 0:
                    print(f"      ... scanned {total_rows_scanned:,} rows")
                    
        runtimes[source_label] = round(time.time() - t_src_start, 2)
        print(f"   {source_label} complete in {runtimes[source_label]}s.")
        del s1_nt_map, s1_at_map
        gc.collect()

    t_global_duration = time.time() - t_global_start

    # Clean up S1 maps to free memory before validation
    del s1_name_map, s1_addr_map, s1_num_map
    gc.collect()

    # ==========================================
    # PILOT RECALL VALIDATION
    # ==========================================
    print("\n3. Validating Block 01 Structural Constraints and Recall...")
    
    pilot_gt: Dict[str, Set[str]] = defaultdict(set)
    gt_reader = pd.read_csv(GROUND_TRUTH_FILE, sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
    for chunk in gt_reader:
        mask = chunk["source1_entity_id"].isin(s1_ids)
        if mask.any():
            for _, row in chunk[mask].iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                raw_matched = str(row.get("matched_entity_ids", "")).strip()
                if raw_matched:
                    pilot_gt[s1_id].update({m.strip() for m in raw_matched.split(",") if m.strip()})

    total_true_matches = sum(len(m) for m in pilot_gt.values())
    recovered_matches = 0
    candidate_counts = defaultdict(int)
    
    # Lightweight structural checks
    duplicate_errors = 0
    prefix_errors = 0
    s1_mismatch_errors = 0
    
    prev_cand_id = None
    seen_s1s_for_cand = set()
    
    with open(OUTPUT_PAIRS_FILE, "r", encoding="utf-8") as f:
        header = next(f, None)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3: continue
            
            s1_id = parts[0]
            cand_id = parts[1]
            src_lbl = parts[2]
            
            candidate_counts[s1_id] += 1
            
            if cand_id != prev_cand_id:
                seen_s1s_for_cand.clear()
                prev_cand_id = cand_id
            
            if s1_id in seen_s1s_for_cand:
                duplicate_errors += 1
            seen_s1s_for_cand.add(s1_id)
            
            if not cand_id.startswith(src_lbl):
                prefix_errors += 1
                
            if s1_id not in s1_ids:
                s1_mismatch_errors += 1
            
            if cand_id in pilot_gt.get(s1_id, set()):
                pilot_gt[s1_id].remove(cand_id)
                recovered_matches += 1

    recall = (recovered_matches / total_true_matches) if total_true_matches > 0 else 0.0
    cand_counts_list = [candidate_counts[s1] for s1 in s1_ids]
    avg_cands = sum(cand_counts_list) / len(s1_ids) if s1_ids else 0.0

    if duplicate_errors == 0 and prefix_errors == 0 and s1_mismatch_errors == 0:
        print("  Block 01 structural validation PASS.")

    file_size_mb = os.path.getsize(OUTPUT_PAIRS_FILE) / (1024 * 1024)
    final_rss = get_memory_usage_mb()

    print("\n================== PILOT VALIDATION RESULTS ==================")
    print(f"  Block 01 S1 count: {len(s1_ids):,}")
    print(f"  S2 runtime: {runtimes.get('S2', 0.0):.2f} s")
    print(f"  S3 runtime: {runtimes.get('S3', 0.0):.2f} s")
    print(f"  Total runtime: {t_global_duration:.2f} s")
    print(f"  Candidate count: {total_pairs_written:,}")
    print(f"  Candidates per S1 (avg): {avg_cands:.2f}")
    print(f"  Output file size: {file_size_mb:.2f} MB")
    print(f"  Block 01 true matches: {total_true_matches:,}")
    print(f"  Block 01 recovered matches: {recovered_matches:,}")
    print(f"  Block 01 candidate recall: {recall:.4%}")
    if final_rss > 0:
        print(f"  Final RSS memory: {final_rss:.2f} MB")
    print("==============================================================\n")
    print(f"Output pairs written to: {OUTPUT_PAIRS_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Fast Candidate Generation")
    parser.add_argument("--mode", choices=["pilot_block"], required=True)
    args = parser.parse_args()

    if args.mode == "pilot_block":
        run_pilot_block()
    else:
        print("Mode not supported yet.")


if __name__ == "__main__":
    main()