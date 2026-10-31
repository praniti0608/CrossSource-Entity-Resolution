from __future__ import annotations

"""
feature_generation_sample.py

Phase 2 Feature Generation (Memory Bounded, Reusable Architecture).
Amazon ML Challenge 2026 - Business Entity Resolution.

Algorithm:
1. Streams candidate_pairs_fast.tsv to split pairs into temporary S2/S3 disk files,
   collecting ONLY the required S1 query IDs in memory (strictly bounded).
2. Streams train_ground_truth.tsv to load matches only for the required S1 entities.
3. Streams S1 TSV to load ONLY the required query records into memory.
4. Processes targets via a Two-Pointer Streaming Merge: 
   - A generator yields candidates from the temporary file grouped by target ID.
   - The target TSV is streamed sequentially. 
   - When the TSV row matches the candidate group's target ID, features are computed 
     and flushed directly to output/features_sample.tsv.
   - Ordering safety checks ensure zero missing or duplicated candidates.
5. Clears memory, then repeats the process for the next source.
"""

import os
import sys
import gc
import time
import ast
import argparse
import unicodedata
import re
import difflib
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple, Iterable
from collections import defaultdict

import pandas as pd

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
CANDIDATE_PAIRS_FILE = OUTPUT_DIR / "candidate_pairs_fast.tsv"
FEATURES_OUTPUT_FILE = OUTPUT_DIR / "features_sample.tsv"


# --- Normalization & Tokenization Utilities ---
def _default_normalize(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return ""
    text_str = unicodedata.normalize("NFKC", str(text))
    text_str = re.sub(r"\s+", " ", text_str).strip().casefold()
    return text_str

def normalize_text_field(text: Optional[str], field_type: str) -> str:
    if text is None or pd.isna(text):
        return ""
    if prep_mod is not None:
        fn_names = [f"normalize_{field_type}", "normalize_text", "clean_text"]
        for fn_name in fn_names:
            if hasattr(prep_mod, fn_name):
                res = getattr(prep_mod, fn_name)(text)
                return "" if res is None or pd.isna(res) else str(res).strip()
    return _default_normalize(text)

def to_alnum(text: str) -> str:
    if not text: return ""
    if prep_mod is not None and hasattr(prep_mod, "to_alnum_representation"):
        res = prep_mod.to_alnum_representation(text)
        return str(res).strip() if res else ""
    
    # Fallback to keep letters/digits
    kept = []
    prev_space = False
    for ch in text:
        if re.match(r"[^\W\d_]|\d", ch, re.UNICODE):
            kept.append(ch)
            prev_space = False
        else:
            if not prev_space and kept:
                kept.append(" ")
                prev_space = True
    return "".join(kept).strip()

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

def extract_numeric_tokens(text: str) -> Set[str]:
    if not text:
        return set()
    return set(re.findall(r"\d+", text))


# --- Data Structures ---
class RecordInfo:
    """Memory-efficient container for essential record attributes."""
    __slots__ = (
        "entity_id",
        "country",
        "name_norm",
        "name_alnum",
        "name_tokens",
        "addr_norm",
        "addr_alnum",
        "addr_tokens",
        "num_tokens",
    )

    def __init__(self, entity_id: str, country: str, name_norm: str, name_alnum: str, 
                 name_tokens: Set[str], addr_norm: str, addr_alnum: str, 
                 addr_tokens: Set[str], num_tokens: Set[str]):
        self.entity_id = entity_id
        self.country = country
        self.name_norm = name_norm
        self.name_alnum = name_alnum
        self.name_tokens = name_tokens
        self.addr_norm = addr_norm
        self.addr_alnum = addr_alnum
        self.addr_tokens = addr_tokens
        self.num_tokens = num_tokens

def extract_record_info(row) -> RecordInfo:
    if hasattr(row, "_asdict"):
        row = row._asdict()
        
    eid = str(row.get("entity_id", "")).strip()
    
    c = row.get("country_normalized")
    if pd.isna(c) or not c: c = normalize_text_field(row.get("country"), "country")
    
    n = row.get("business_name_normalized")
    if pd.isna(n) or not n: n = normalize_text_field(row.get("business_name"), "business_name")
        
    a = row.get("business_address_normalized")
    if pd.isna(a) or not a: a = normalize_text_field(row.get("business_address"), "business_address")
        
    c, n, a = str(c).strip(), str(n).strip(), str(a).strip()
    n_alnum = to_alnum(n)
    a_alnum = to_alnum(a)
    
    n_toks = safe_parse_tokens(row.get("business_name_tokens", None), n)
    a_toks = safe_parse_tokens(row.get("business_address_tokens", None), a)
    
    addr_raw = str(row.get("business_address", ""))
    num_toks = extract_numeric_tokens(addr_raw if addr_raw.strip() else a)
        
    return RecordInfo(eid, c, n, n_alnum, n_toks, a, a_alnum, a_toks, num_toks)


# --- Feature Calculators ---
def char_similarity(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def containment_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / min(len(set1), len(set2))


# --- Streaming Helpers ---
def iter_candidate_blocks(file_path: Path) -> Iterable[Tuple[str, List[List[str]]]]:
    """Yields candidate pairs grouped by candidate_entity_id."""
    current_target_id = None
    block = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 2: continue
            
            target_id = parts[1]
            if current_target_id is None:
                current_target_id = target_id
                block.append(parts)
            elif target_id == current_target_id:
                block.append(parts)
            else:
                yield current_target_id, block
                current_target_id = target_id
                block = [parts]
                
        if block:
            yield current_target_id, block


# --- Core Execution ---
def run_feature_generation():
    # Enforce safe terminal output encoding
    if hasattr(sys.stdout, 'reconfigure'):
        try: sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception: pass
    
    print("\n=======================================================")
    print("PHASE 2: MEMORY-BOUNDED FEATURE GENERATION (TWO-POINTER MERGE)")
    print("=======================================================")
    
    if not CANDIDATE_PAIRS_FILE.exists():
        raise FileNotFoundError(f"Missing {CANDIDATE_PAIRS_FILE}. Run Phase 1 first.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    with tempfile.TemporaryDirectory(prefix="featgen_") as tmp_dir:
        tmp_s2_file = Path(tmp_dir) / "s2_candidates.tmp"
        tmp_s3_file = Path(tmp_dir) / "s3_candidates.tmp"
        
        # --- STEP 1: Partition Candidates to Disk & Extract S1 IDs ---
        print(f"\n[Step 1/4] Partitioning candidates and isolating required S1 IDs...")
        needed_s1_ids = set()
        total_pairs = 0
        
        with open(CANDIDATE_PAIRS_FILE, "r", encoding="utf-8") as f_in, \
             open(tmp_s2_file, "w", encoding="utf-8") as f_s2, \
             open(tmp_s3_file, "w", encoding="utf-8") as f_s3:
            
            next(f_in, None) # Skip header
            for line in f_in:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    s1_id, cand_id = parts[0], parts[1]
                    needed_s1_ids.add(s1_id)
                    total_pairs += 1
                    
                    if cand_id.startswith("S2-"):
                        f_s2.write(line)
                    elif cand_id.startswith("S3-"):
                        f_s3.write(line)

        print(f"  Total Candidate Pairs: {total_pairs:,}")
        print(f"  Required Unique S1: {len(needed_s1_ids):,}")

        # --- STEP 2: Extract Labels from Ground Truth ---
        print("\n[Step 2/4] Parsing Ground Truth for validation labels...")
        gt_map: Dict[str, Set[str]] = defaultdict(set)
        gt_reader = pd.read_csv(GROUND_TRUTH_FILE, sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
        for chunk in gt_reader:
            mask = chunk["source1_entity_id"].isin(needed_s1_ids)
            if mask.any():
                for _, row in chunk[mask].iterrows():
                    s1_id = str(row["source1_entity_id"]).strip()
                    raw_matched = str(row.get("matched_entity_ids", "")).strip()
                    if raw_matched:
                        gt_map[s1_id].update({m.strip() for m in raw_matched.split(",") if m.strip()})
        
        # --- STEP 3: Load Needed S1 Records ---
        print("\n[Step 3/4] Loading required S1 records into memory...")
        s1_records: Dict[str, RecordInfo] = {}
        s1_found = 0
        
        reader = pd.read_csv(TRAIN_SOURCE_FILES["source1"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
        for chunk in reader:
            mask = chunk["entity_id"].isin(needed_s1_ids)
            if mask.any():
                for _, row in chunk[mask].iterrows():
                    rec = extract_record_info(row)
                    s1_records[rec.entity_id] = rec
                    s1_found += 1
            if s1_found >= len(needed_s1_ids):
                break
        print(f"  Loaded {s1_found:,} S1 records.")

        # --- Setup Output File ---
        headers = [
            "source1_entity_id", "candidate_entity_id", "label",
            "name_exact_match", "name_char_similarity", "name_jaccard", "name_containment",
            "name_shared_token_count", "name_token_count_diff", "name_len_diff",
            "addr_exact_match", "addr_char_similarity", "addr_jaccard", "addr_containment",
            "addr_shared_token_count", "addr_token_count_diff", "addr_len_diff",
            "addr_shared_numeric_count", "country_exact_match",
            "cg_exact_name", "cg_exact_address", "cg_name_token", "cg_address_token"
        ]
        with open(FEATURES_OUTPUT_FILE, "w", encoding="utf-8") as fout:
            fout.write("\t".join(headers) + "\n")

        # Global tracking stats
        stats_positive = 0
        stats_negative = 0
        stats_missing_s1 = 0
        total_processed = 0

        # --- STEP 4: Two-Pointer Streaming Merge for Targets ---
        print("\n[Step 4/4] Computing features via Two-Pointer Streaming Merge...")
        
        for source_label, path, tmp_cand_path in [
            ("S2", TRAIN_SOURCE_FILES["source2"], tmp_s2_file),
            ("S3", TRAIN_SOURCE_FILES["source3"], tmp_s3_file)
        ]:
            print(f"\n  Processing {source_label}...")
            
            if os.path.getsize(tmp_cand_path) == 0:
                print(f"    No candidates for {source_label}, skipping.")
                continue
                
            cand_iter = iter_candidate_blocks(tmp_cand_path)
            try:
                candidate_target_id, cand_block = next(cand_iter)
            except StopIteration:
                print(f"    No candidates for {source_label}, skipping.")
                continue

            target_reader = pd.read_csv(path, sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
            
            source_processed = 0
            source_positive = 0
            source_negative = 0
            
            processed_targets = set()
            last_target_id = None
            is_exhausted = False
            
            with open(FEATURES_OUTPUT_FILE, "a", encoding="utf-8") as fout:
                batch_lines = []
                
                for chunk in target_reader:
                    if is_exhausted:
                        break
                        
                    for row in chunk.itertuples(index=False):
                        if is_exhausted:
                            break
                            
                        row_eid = str(getattr(row, "entity_id", "")).strip()
                        if not row_eid: 
                            continue
                            
                        if row_eid == candidate_target_id:
                            if candidate_target_id in processed_targets:
                                raise RuntimeError(f"Candidate ordering does not match target TSV ordering. Target ID {candidate_target_id} appeared earlier.")
                            
                            processed_targets.add(candidate_target_id)
                            last_target_id = candidate_target_id
                            
                            tgt_rec = extract_record_info(row)
                            
                            for cand_parts in cand_block:
                                s1_id = cand_parts[0]
                                cand_id = cand_parts[1]
                                methods_str = cand_parts[3] if len(cand_parts) >= 4 else ""
                                
                                s1 = s1_records.get(s1_id)
                                if not s1: stats_missing_s1 += 1
                                
                                label = 1 if cand_id in gt_map.get(s1_id, set()) else 0
                                
                                source_processed += 1
                                if label == 1:
                                    source_positive += 1
                                else:
                                    source_negative += 1
                                    
                                if not s1:
                                    row_data = [s1_id, cand_id, str(label)] + ["0"] * (len(headers) - 3)
                                    batch_lines.append("\t".join(row_data) + "\n")
                                    continue
                                    
                                # Name features
                                name_exact = 1 if s1.name_norm and tgt_rec.name_norm and s1.name_norm == tgt_rec.name_norm else 0
                                name_sim = char_similarity(s1.name_alnum, tgt_rec.name_alnum)
                                name_jac = jaccard_similarity(s1.name_tokens, tgt_rec.name_tokens)
                                name_cont = containment_similarity(s1.name_tokens, tgt_rec.name_tokens)
                                name_shared = len(s1.name_tokens & tgt_rec.name_tokens)
                                name_diff = abs(len(s1.name_tokens) - len(tgt_rec.name_tokens))
                                name_len_diff = abs(len(s1.name_norm) - len(tgt_rec.name_norm))
                                
                                # Address features
                                addr_exact = 1 if s1.addr_norm and tgt_rec.addr_norm and s1.addr_norm == tgt_rec.addr_norm else 0
                                addr_sim = char_similarity(s1.addr_alnum, tgt_rec.addr_alnum)
                                addr_jac = jaccard_similarity(s1.addr_tokens, tgt_rec.addr_tokens)
                                addr_cont = containment_similarity(s1.addr_tokens, tgt_rec.addr_tokens)
                                addr_shared = len(s1.addr_tokens & tgt_rec.addr_tokens)
                                addr_diff = abs(len(s1.addr_tokens) - len(tgt_rec.addr_tokens))
                                addr_len_diff = abs(len(s1.addr_norm) - len(tgt_rec.addr_norm))
                                addr_num = len(s1.num_tokens & tgt_rec.num_tokens)
                                
                                # Other features
                                c_exact = 1 if s1.country and tgt_rec.country and s1.country == tgt_rec.country else 0
                                
                                methods = set(methods_str.split("|"))
                                cg_en = 1 if "exact_name" in methods else 0
                                cg_ea = 1 if "exact_address" in methods else 0
                                cg_nt = 1 if "name_token" in methods else 0
                                cg_at = 1 if "address_token" in methods else 0
                                
                                row_data = [
                                    s1_id, cand_id, str(label),
                                    str(name_exact), f"{name_sim:.4f}", f"{name_jac:.4f}", f"{name_cont:.4f}",
                                    str(name_shared), str(name_diff), str(name_len_diff),
                                    str(addr_exact), f"{addr_sim:.4f}", f"{addr_jac:.4f}", f"{addr_cont:.4f}",
                                    str(addr_shared), str(addr_diff), str(addr_len_diff), str(addr_num),
                                    str(c_exact),
                                    str(cg_en), str(cg_ea), str(cg_nt), str(cg_at)
                                ]
                                
                                batch_lines.append("\t".join(row_data) + "\n")
                                if len(batch_lines) >= 10_000:
                                    fout.writelines(batch_lines)
                                    batch_lines.clear()
                                    
                            try:
                                candidate_target_id, cand_block = next(cand_iter)
                            except StopIteration:
                                is_exhausted = True
                                break
                                
                if batch_lines:
                    fout.writelines(batch_lines)
                    batch_lines.clear()
                    
            if not is_exhausted and candidate_target_id is not None:
                raise RuntimeError(f"Target stream exhausted but candidates remain. Leftover target ID: {candidate_target_id}")

            print(f"  Processed {source_processed:,} candidate pairs for {source_label}.")
            print(f"  Positive labels: {source_positive:,}")
            print(f"  Negative labels: {source_negative:,}")

            total_processed += source_processed
            stats_positive += source_positive
            stats_negative += source_negative
            
            # Memory reset per source
            processed_targets.clear()
            gc.collect()

    total_time = time.time() - t0
    pos_rate = (stats_positive / total_processed) * 100 if total_processed else 0.0

    print("\n================== FEATURE GENERATION SUMMARY ==================")
    print(f"  Total Candidate Pairs Processed : {total_processed:,}")
    print(f"  Positive Pairs (Label=1)        : {stats_positive:,}")
    print(f"  Negative Pairs (Label=0)        : {stats_negative:,}")
    print(f"  Positive Class Rate             : {pos_rate:.2f}%")
    print(f"  Missing S1 Records              : {stats_missing_s1:,}")
    print(f"  Total Runtime                   : {total_time:.2f} seconds")
    print(f"\n  Output TSV Path: {FEATURES_OUTPUT_FILE}")
    print("================================================================")


if __name__ == "__main__":
    run_feature_generation()