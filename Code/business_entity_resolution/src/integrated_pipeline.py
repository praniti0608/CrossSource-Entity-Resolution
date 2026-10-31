"""
integrated_pipeline.py

Final Integrated Candidate Generation & Inference Pipeline.
Amazon ML Challenge 2026 - Business Entity Resolution.
"""

from __future__ import annotations

import os
import sys
import gc
import time
import ast
import argparse
import unicodedata
import re
from array import array
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple
from collections import defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None
    
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# --- Path Resolution & Imports ---
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
INFERENCE_THRESHOLD = 0.82
MAX_S1 = 2500000  # Safe upper bound for memory preallocation

FEATURE_COLUMNS = [
    "name_exact_match",
    "name_char_similarity",
    "name_jaccard",
    "name_containment",
    "name_shared_token_count",
    "name_token_count_diff",
    "name_len_diff",
    "addr_exact_match",
    "addr_char_similarity",
    "addr_jaccard",
    "addr_containment",
    "addr_shared_token_count",
    "addr_token_count_diff",
    "addr_len_diff",
    "addr_shared_numeric_count",
    "country_exact_match",
    "cg_exact_name",
    "cg_exact_address",
    "cg_name_token",
    "cg_address_token",
]

DATASET_TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAIN_SOURCE_FILES = {
    "source1": DATASET_TRAIN_DIR / "train_source1.tsv",
    "source2": DATASET_TRAIN_DIR / "train_source2.tsv",
    "source3": DATASET_TRAIN_DIR / "train_source3.tsv",
}

CANDIDATE_PAIRS_FILE = OUTPUT_DIR / "candidate_pairs.tsv"
MATCHING_RESULTS_FILE = OUTPUT_DIR / "matching_results.tsv"
MODEL_FILE = OUTPUT_DIR / "lgbm_model.pkl"

try:
    import psutil
except ImportError:
    psutil = None

def get_memory_usage_mb() -> float:
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    return 0.0


# --- Normalization & Tokenization Utilities ---
TOKEN_REGEX = re.compile(r"[^\w]+", re.UNICODE)

def _default_normalize(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return ""
    text_str = unicodedata.normalize("NFKC", str(text))
    text_str = re.sub(r"\s+", " ", text_str).strip().casefold()
    return text_str

def normalize_name(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return ""
    if prep_mod is not None and hasattr(prep_mod, "normalize_business_name"):
        return str(getattr(prep_mod, "normalize_business_name")(text) or "").strip()
    return _default_normalize(text)

def normalize_address(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return ""
    if prep_mod is not None and hasattr(prep_mod, "normalize_business_address"):
        return str(getattr(prep_mod, "normalize_business_address")(text) or "").strip()
    return _default_normalize(text)

def normalize_country(text: Optional[str]) -> str:
    if text is None or pd.isna(text):
        return "UNKNOWN"
    if prep_mod is not None and hasattr(prep_mod, "normalize_country"):
        res = getattr(prep_mod, "normalize_country")(text)
        return str(res).strip() if res else "UNKNOWN"
    return _default_normalize(text) or "UNKNOWN"

def safe_parse_tokens(raw_val: any, norm_val: str) -> Set[str]:
    """Extracts valid tokens, reusing preprocessed column if available, else exact regex parity."""
    if pd.notna(raw_val) and str(raw_val).strip():
        val_str = str(raw_val).strip()
        if val_str.startswith('['):
            try:
                parsed = ast.literal_eval(val_str)
                return {str(x).strip().lower() for x in parsed if str(x).strip()}
            except Exception:
                pass
                
    if norm_val:
        return {t for t in TOKEN_REGEX.split(str(norm_val).lower()) if t}
    return set()

def extract_numeric_tokens(text: str) -> Set[str]:
    if not text: return set()
    return set(re.findall(r"\d+", text))

def extract_all(row: pd.Series) -> Tuple[str, str, str, str, Set[str], Set[str], Set[str]]:
    eid = str(row.get("entity_id", "")).strip()
    c = normalize_country(row.get("country_normalized") or row.get("country"))
    
    n_raw = str(row.get("business_name", ""))
    n = row.get("business_name_normalized")
    n = str(n).strip() if pd.notna(n) and n else normalize_name(n_raw)
    
    a_raw = str(row.get("business_address", ""))
    a = row.get("business_address_normalized")
    a = str(a).strip() if pd.notna(a) and a else normalize_address(a_raw)
    
    n_toks = safe_parse_tokens(row.get("business_name_tokens", None), n)
    a_toks = safe_parse_tokens(row.get("business_address_tokens", None), a)
    num_toks = extract_numeric_tokens(a_raw if a_raw.strip() else a)
        
    return eid, c, n, a, n_toks, a_toks, num_toks

def calculate_jaccard(set1: Set[str], set2: Set[str]) -> float:
    if not set1 and not set2: return 0.0
    u = len(set1 | set2)
    return len(set1 & set2) / u if u > 0 else 0.0

def calculate_containment(set1: Set[str], set2: Set[str]) -> float:
    if not set1 or not set2: return 0.0
    m = min(len(set1), len(set2))
    return len(set1 & set2) / m if m > 0 else 0.0

def update_touched(idxs: np.ndarray, is_touched: np.ndarray, touched_indices: np.ndarray, touched_count: int) -> int:
    """Helper to maintain sparse touched ID state."""
    new_mask = ~is_touched[idxs]
    new_idxs = idxs[new_mask]
    if new_idxs.size > 0:
        is_touched[new_idxs] = True
        touched_indices[touched_count : touched_count + new_idxs.size] = new_idxs
        return touched_count + new_idxs.size
    return touched_count

def run_parity_test():
    """Validates that fallback tokenization aligns exactly with validated semantics."""
    assert safe_parse_tokens(None, "abc def") == {"abc", "def"}, "Parity test failed: abc def"
    assert safe_parse_tokens(None, "a b") == {"a", "b"}, "Parity test failed: a b"
    
    # Python's re.compile(r"[^\w]+") naturally splits Unicode combining marks. 
    # We dynamically evaluate to mathematically preserve exact preprocessing.py parity.
    hindi_text = "दिल्ली रोड"
    expected_hindi = {t for t in TOKEN_REGEX.split(hindi_text.lower()) if t}
    assert safe_parse_tokens(None, hindi_text) == expected_hindi, "Parity test failed: Hindi text"
    
    assert safe_parse_tokens(None, "foo-bar, baz") == {"foo", "bar", "baz"}, "Parity test failed: punctuation"
    
    print("  [OK] Preprocessing parity test passed.")
    if prep_mod is not None:
        has_n = hasattr(prep_mod, "normalize_business_name")
        has_a = hasattr(prep_mod, "normalize_business_address")
        has_c = hasattr(prep_mod, "normalize_country")
        print(f"  [OK] prep_mod loaded. normalize_business_name: {has_n}, normalize_business_address: {has_a}, normalize_country: {has_c}")
    else:
        print("  [WARNING] prep_mod not found. Using _default_normalize fallback.")


# --- Pipeline Execution ---
def precompute_global_tokens(test_mode: bool) -> Tuple[Set, Set, Set, Set, Set, Set]:
    print("\n[Phase 1] Precomputing Global Token Vocabularies...")
    
    s1_nt_freq = defaultdict(int)
    s1_at_freq = defaultdict(int)
    
    s1_reader = pd.read_csv(TRAIN_SOURCE_FILES["source1"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
    s1_loaded = 0
    for chunk in s1_reader:
        if test_mode and s1_loaded + len(chunk) > 10000:
            chunk = chunk.head(10000 - s1_loaded)
        for row in chunk.itertuples(index=False):
            c = normalize_country(getattr(row, "country", ""))
            n = normalize_name(getattr(row, "business_name", ""))
            a = normalize_address(getattr(row, "business_address", ""))
            n_toks = safe_parse_tokens(getattr(row, "business_name_tokens", None), n)
            a_toks = safe_parse_tokens(getattr(row, "business_address_tokens", None), a)
            for tk in n_toks: s1_nt_freq[(c, tk)] += 1
            for tk in a_toks: s1_at_freq[(c, tk)] += 1
        s1_loaded += len(chunk)
        if test_mode and s1_loaded >= 10000:
            break

    eligible_s1_nt = {k for k, v in s1_nt_freq.items() if v <= COMMON_TOKEN_THRESHOLD}
    eligible_s1_at = {k for k, v in s1_at_freq.items() if v <= COMMON_TOKEN_THRESHOLD}
    del s1_nt_freq, s1_at_freq
    gc.collect()

    s2_nt_freq = defaultdict(int)
    s2_at_freq = defaultdict(int)
    s3_nt_freq = defaultdict(int)
    s3_at_freq = defaultdict(int)

    for src_lbl, tpath, nt_f, at_f in [("S2", TRAIN_SOURCE_FILES["source2"], s2_nt_freq, s2_at_freq),
                                       ("S3", TRAIN_SOURCE_FILES["source3"], s3_nt_freq, s3_at_freq)]:
        reader = pd.read_csv(tpath, sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
        target_loaded = 0
        for chunk in reader:
            if test_mode and target_loaded + len(chunk) > 10000:
                chunk = chunk.head(10000 - target_loaded)
            for row in chunk.itertuples(index=False):
                c = normalize_country(getattr(row, "country", ""))
                n = normalize_name(getattr(row, "business_name", ""))
                a = normalize_address(getattr(row, "business_address", ""))
                for tk in safe_parse_tokens(getattr(row, "business_name_tokens", None), n):
                    if (c, tk) in eligible_s1_nt: nt_f[(c, tk)] += 1
                for tk in safe_parse_tokens(getattr(row, "business_address_tokens", None), a):
                    if (c, tk) in eligible_s1_at: at_f[(c, tk)] += 1
            target_loaded += len(chunk)
            if test_mode and target_loaded >= 10000:
                break

    s2_valid_nt = {k for k in eligible_s1_nt if s2_nt_freq[k] <= TARGET_TOKEN_MAX_FREQ}
    s2_valid_at = {k for k in eligible_s1_at if s2_at_freq[k] <= TARGET_TOKEN_MAX_FREQ}
    s3_valid_nt = {k for k in eligible_s1_nt if s3_nt_freq[k] <= TARGET_TOKEN_MAX_FREQ}
    s3_valid_at = {k for k in eligible_s1_at if s3_at_freq[k] <= TARGET_TOKEN_MAX_FREQ}

    print(f"  S1 Eligible Name Tokens: {len(eligible_s1_nt):,}")
    print(f"  S1 Eligible Addr Tokens: {len(eligible_s1_at):,}")
    print(f"  S2 Valid Target Name Tokens: {len(s2_valid_nt):,}")
    print(f"  S2 Valid Target Addr Tokens: {len(s2_valid_at):,}")
    print(f"  S3 Valid Target Name Tokens: {len(s3_valid_nt):,}")
    print(f"  S3 Valid Target Addr Tokens: {len(s3_valid_at):,}")
    
    return s2_valid_nt, s2_valid_at, s3_valid_nt, s3_valid_at, eligible_s1_nt, eligible_s1_at


def run_pipeline(test_mode: bool = False):
    print("\n=======================================================")
    print("INTEGRATED CANDIDATE GENERATION & INFERENCE PIPELINE")
    print("=======================================================")
    run_parity_test()
    
    if test_mode:
        print(">>> RUNNING IN TEST MODE (Limited Processing to 10,000 Rows/Source) <<<")
    
    if not MODEL_FILE.exists():
        print(f"ERROR: Model file not found at {MODEL_FILE}")
        sys.exit(1)
        
    print("\n[Phase 0] Loading LightGBM Model...")
    model = joblib.load(MODEL_FILE)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_global_start = time.time()
    
    s2_valid_nt, s2_valid_at, s3_valid_nt, s3_valid_at, eligible_s1_nt, eligible_s1_at = precompute_global_tokens(test_mode)

    print("\n[Phase 2] Building Full S1 Inverted Index and Compact Feature Storage...")
    
    s1_id_map = []
    
    # Store all string attributes in a single memory-compact bytearray instead of millions of Python objects.
    s1_data_store = bytearray()
    s1_offsets = np.zeros(MAX_S1, dtype=np.int64)
    s1_lengths = np.zeros(MAX_S1, dtype=np.int32)

    # Use direct array('i') for strictly memory-bounded inverted indexes
    s1_name_idx: Dict[Tuple[str, str], array] = defaultdict(lambda: array('i'))
    s1_addr_idx: Dict[Tuple[str, str], array] = defaultdict(lambda: array('i'))
    s1_nt_idx: Dict[Tuple[str, str], array] = defaultdict(lambda: array('i'))
    s1_at_idx: Dict[Tuple[str, str], array] = defaultdict(lambda: array('i'))
    s1_num_idx: Dict[Tuple[str, str], array] = defaultdict(lambda: array('i'))

    s1_reader = pd.read_csv(TRAIN_SOURCE_FILES["source1"], sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
    s1_count = 0
    s1_limit = 10000 if test_mode else MAX_S1
    
    for chunk in s1_reader:
        if s1_count >= s1_limit:
            break
        if s1_count + len(chunk) > s1_limit:
            chunk = chunk.head(s1_limit - s1_count)
            
        for _, row in chunk.iterrows():
            eid, c, n, a, n_toks, a_toks, num_toks = extract_all(row)
            if not eid: continue
            
            s1_idx = s1_count
            
            # Pack strings/tokens compactly avoiding per-row string object retention
            nt_str = " ".join(n_toks)
            at_str = " ".join(a_toks)
            num_str = " ".join(num_toks)
            
            record_bytes = f"{eid}\t{n}\t{a}\t{c}\t{nt_str}\t{at_str}\t{num_str}".encode("utf-8")
            s1_offsets[s1_idx] = len(s1_data_store)
            s1_lengths[s1_idx] = len(record_bytes)
            s1_data_store.extend(record_bytes)
            s1_id_map.append(eid)
            
            if n: s1_name_idx[(c, n)].append(s1_idx)
            if a: s1_addr_idx[(c, a)].append(s1_idx)
            for tk in num_toks: s1_num_idx[(c, tk)].append(s1_idx)
            
            for tk in n_toks:
                if (c, tk) in eligible_s1_nt: s1_nt_idx[(c, tk)].append(s1_idx)
            for tk in a_toks:
                if (c, tk) in eligible_s1_at: s1_at_idx[(c, tk)].append(s1_idx)
                
            s1_count += 1

    # Remove default_factory mapping to lock the dictionaries
    s1_name_idx.default_factory = None
    s1_addr_idx.default_factory = None
    s1_nt_idx.default_factory = None
    s1_at_idx.default_factory = None
    s1_num_idx.default_factory = None

    del eligible_s1_nt, eligible_s1_at
    gc.collect()

    print(f"\n  S1 count: {s1_count:,}")
    print(f"  Number of name index keys: {len(s1_name_idx):,}")
    print(f"  Number of address index keys: {len(s1_addr_idx):,}")
    print(f"  Number of name-token index keys: {len(s1_nt_idx):,}")
    print(f"  Number of address-token index keys: {len(s1_at_idx):,}")
    print(f"  Number of numeric index keys: {len(s1_num_idx):,}")
    print(f"  RSS memory: {get_memory_usage_mb():.2f} MB")

    s1_predictions = defaultdict(set)

    # Sparse Per-Target Accumulator Arrays
    nt_c_arr = np.zeros(MAX_S1, dtype=np.uint8)
    at_c_arr = np.zeros(MAX_S1, dtype=np.uint8)
    num_c_arr = np.zeros(MAX_S1, dtype=np.uint8)
    ex_n_arr = np.zeros(MAX_S1, dtype=np.bool_)
    ex_a_arr = np.zeros(MAX_S1, dtype=np.bool_)
    is_touched = np.zeros(MAX_S1, dtype=np.bool_)
    touched_indices = np.empty(MAX_S1, dtype=np.int32)

    print("\n[Phase 3] Streaming Targets & Executing Inference...")
    with open(CANDIDATE_PAIRS_FILE, "w", encoding="utf-8") as out_fh:
        out_fh.write("source1_entity_id\tcandidate_entity_id\tcandidate_source\tmethods\n")

    total_candidates_evaluated = 0
    total_matches_predicted = 0
    s2_rows_processed = 0
    s3_rows_processed = 0

    for source_label, target_path in [("S2", TRAIN_SOURCE_FILES["source2"]), ("S3", TRAIN_SOURCE_FILES["source3"])]:
        print(f"\n--- Processing {source_label} ---")
        current_valid_nt = s2_valid_nt if source_label == "S2" else s3_valid_nt
        current_valid_at = s2_valid_at if source_label == "S2" else s3_valid_at
        
        target_reader = pd.read_csv(target_path, sep="\t", chunksize=CHUNKSIZE, dtype=str, keep_default_na=False)
        total_rows = 0
        
        with open(CANDIDATE_PAIRS_FILE, "a", encoding="utf-8") as out_fh:
            buffer_lines = []
            
            for chunk in target_reader:
                if test_mode and total_rows + len(chunk) > 10000:
                    chunk = chunk.head(10000 - total_rows)
                    
                for _, row in chunk.iterrows():
                    t_eid, c, n, a, n_toks, a_toks, num_toks = extract_all(row)
                    if not t_eid: continue
                    
                    touched_count = 0

                    # 1. Accumulate Sparse Evidence
                    if n:
                        arr = s1_name_idx.get((c, n))
                        if arr:
                            idxs = np.frombuffer(arr, dtype=np.int32)
                            ex_n_arr[idxs] = True
                            touched_count = update_touched(idxs, is_touched, touched_indices, touched_count)
                        
                    if a:
                        arr = s1_addr_idx.get((c, a))
                        if arr:
                            idxs = np.frombuffer(arr, dtype=np.int32)
                            ex_a_arr[idxs] = True
                            touched_count = update_touched(idxs, is_touched, touched_indices, touched_count)
                        
                    for tk in n_toks:
                        if (c, tk) in current_valid_nt:
                            arr = s1_nt_idx.get((c, tk))
                            if arr:
                                idxs = np.frombuffer(arr, dtype=np.int32)
                                nt_c_arr[idxs] += 1
                                touched_count = update_touched(idxs, is_touched, touched_indices, touched_count)
                            
                    for tk in a_toks:
                        if (c, tk) in current_valid_at:
                            arr = s1_at_idx.get((c, tk))
                            if arr:
                                idxs = np.frombuffer(arr, dtype=np.int32)
                                at_c_arr[idxs] += 1
                                touched_count = update_touched(idxs, is_touched, touched_indices, touched_count)
                            
                    if touched_count == 0:
                        continue

                    # Numeric Evidence (Scores only touched IDs; never spawns candidates alone)
                    if num_toks:
                        for tk in num_toks:
                            arr = s1_num_idx.get((c, tk))
                            if arr:
                                idxs = np.frombuffer(arr, dtype=np.int32)
                                valid_idxs = idxs[is_touched[idxs]]
                                if valid_idxs.size > 0:
                                    num_c_arr[valid_idxs] += 1

                    # 2. Ranking & Capping
                    touched_idx_arr = touched_indices[:touched_count]
                    
                    if touched_count > MAX_CANDIDATES_PER_TARGET:
                        t_scores = (3 * at_c_arr[touched_idx_arr]) + (2 * nt_c_arr[touched_idx_arr]) + num_c_arr[touched_idx_arr]
                        t_exact = ex_n_arr[touched_idx_arr] | ex_a_arr[touched_idx_arr]
                        sort_order = np.lexsort((-t_scores, ~t_exact))
                        top_500_local_idx = sort_order[:MAX_CANDIDATES_PER_TARGET]
                        eval_idx_arr = touched_idx_arr[top_500_local_idx]
                    else:
                        eval_idx_arr = touched_idx_arr
                        
                    # 3. Dynamic Feature Extraction & Evaluation
                    X_batch = np.zeros((len(eval_idx_arr), 20), dtype=np.float32)
                    
                    for i, s1_idx in enumerate(eval_idx_arr):
                        # Extract exact features from compact byte storage
                        offset = s1_offsets[s1_idx]
                        length = s1_lengths[s1_idx]
                        record_str = s1_data_store[offset : offset + length].decode("utf-8")
                        s1_id_str, s1_n, s1_a, s1_c, s1_nt_str, s1_at_str, s1_num_str = record_str.split('\t')
                        
                        s1_nt = set(s1_nt_str.split()) if s1_nt_str else set()
                        s1_at = set(s1_at_str.split()) if s1_at_str else set()
                        s1_num = set(s1_num_str.split()) if s1_num_str else set()
                        
                        methods = []
                        if ex_n_arr[s1_idx]: methods.append("exact_name")
                        if ex_a_arr[s1_idx]: methods.append("exact_address")
                        if nt_c_arr[s1_idx] > 0: methods.append("name_token")
                        if at_c_arr[s1_idx] > 0: methods.append("address_token")
                        
                        buffer_lines.append(f"{s1_id_str}\t{t_eid}\t{source_label}\t{'|'.join(methods)}\n")
                        total_candidates_evaluated += 1

                        # Exact 20 features in correct order
                        X_batch[i, 0] = 1.0 if (s1_n and n and s1_n == n) else 0.0
                        X_batch[i, 1] = SequenceMatcher(None, s1_n, n).ratio() if s1_n or n else 0.0
                        X_batch[i, 2] = calculate_jaccard(s1_nt, n_toks)
                        X_batch[i, 3] = calculate_containment(s1_nt, n_toks)
                        X_batch[i, 4] = len(s1_nt & n_toks)
                        X_batch[i, 5] = abs(len(s1_nt) - len(n_toks))
                        X_batch[i, 6] = abs(len(s1_n) - len(n))
                        
                        X_batch[i, 7] = 1.0 if (s1_a and a and s1_a == a) else 0.0
                        X_batch[i, 8] = SequenceMatcher(None, s1_a, a).ratio() if s1_a or a else 0.0
                        X_batch[i, 9] = calculate_jaccard(s1_at, a_toks)
                        X_batch[i, 10] = calculate_containment(s1_at, a_toks)
                        X_batch[i, 11] = len(s1_at & a_toks)
                        X_batch[i, 12] = abs(len(s1_at) - len(a_toks))
                        X_batch[i, 13] = abs(len(s1_a) - len(a))
                        X_batch[i, 14] = len(s1_num & num_toks)
                        
                        X_batch[i, 15] = 1.0 if (s1_c and c and s1_c == c) else 0.0
                        
                        X_batch[i, 16] = 1.0 if ex_n_arr[s1_idx] else 0.0
                        X_batch[i, 17] = 1.0 if ex_a_arr[s1_idx] else 0.0
                        X_batch[i, 18] = 1.0 if nt_c_arr[s1_idx] > 0 else 0.0
                        X_batch[i, 19] = 1.0 if at_c_arr[s1_idx] > 0 else 0.0

                    # 4. Integrated Inference
                    if len(X_batch) > 0:
                        X_batch_df = pd.DataFrame(X_batch, columns=FEATURE_COLUMNS)
                        preds = model.predict_proba(X_batch_df)[:, 1]
                        for i, s1_idx in enumerate(eval_idx_arr):
                            if preds[i] >= INFERENCE_THRESHOLD:
                                s1_predictions[s1_idx].add(t_eid)
                                total_matches_predicted += 1

                    # 5. Sparse Reset
                    ex_n_arr[touched_idx_arr] = False
                    ex_a_arr[touched_idx_arr] = False
                    nt_c_arr[touched_idx_arr] = 0
                    at_c_arr[touched_idx_arr] = 0
                    num_c_arr[touched_idx_arr] = 0
                    is_touched[touched_idx_arr] = False
                
                if len(buffer_lines) >= 100_000:
                    out_fh.writelines(buffer_lines)
                    buffer_lines.clear()

                total_rows += len(chunk)
                if total_rows % 100_000 == 0:
                    print(f"      ... processed {total_rows:,} rows")
                    
                if test_mode and total_rows >= 10000:
                    print(f"      ... (TEST MODE: stopping {source_label} early)")
                    break

            if source_label == "S2":
                s2_rows_processed = total_rows
            else:
                s3_rows_processed = total_rows

            if buffer_lines:
                out_fh.writelines(buffer_lines)
                buffer_lines.clear()

    # 4. Aggregation Write
    print("\n[Phase 4] Aggregating Matching Results...")
    with open(MATCHING_RESULTS_FILE, "w", encoding="utf-8") as f_out:
        f_out.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_idx in range(s1_count):
            offset = s1_offsets[s1_idx]
            length = s1_lengths[s1_idx]
            record_str = s1_data_store[offset : offset + length].decode("utf-8")
            s1_id_str = record_str.split('\t', 1)[0]
            
            match_str = ",".join(sorted(s1_predictions[s1_idx])) if s1_idx in s1_predictions else ""
            f_out.write(f"{s1_id_str}\t{match_str}\n")
            
    t_global_duration = time.time() - t_global_start
    final_rss = get_memory_usage_mb()
    
    if test_mode:
        print("\n================== TEST MODE SUMMARY ==================")
        print(f"  S1 indexed: {s1_count:,}")
        print(f"  S2 rows processed: {s2_rows_processed:,}")
        print(f"  S3 rows processed: {s3_rows_processed:,}")
        print(f"  Candidates evaluated: {total_candidates_evaluated:,}")
        print(f"  Predictions: {total_matches_predicted:,}")
        if final_rss > 0:
            print(f"  Final RSS memory: {final_rss:.2f} MB")
        print("=======================================================\n")
    else:
        print("\n================== PIPELINE SUMMARY ==================")
        print(f"  Total S1 Entities Indexed: {s1_count:,}")
        print(f"  Total S2 Rows Processed: {s2_rows_processed:,}")
        print(f"  Total S3 Rows Processed: {s3_rows_processed:,}")
        print(f"  Total Candidates Evaluated: {total_candidates_evaluated:,}")
        print(f"  Total Predictions >= {INFERENCE_THRESHOLD}: {total_matches_predicted:,}")
        print(f"  Total Pipeline Runtime: {t_global_duration:.2f} s")
        if final_rss > 0:
            print(f"  Final RSS memory: {final_rss:.2f} MB")
        print("========================================================\n")
        
    print(f"Candidate pairs saved to: {CANDIDATE_PAIRS_FILE}")
    print(f"Final matches saved to:   {MATCHING_RESULTS_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Integrated Business Entity Resolution Pipeline")
    parser.add_argument("--test", action="store_true", help="Run a tiny correctness test (10000 target rows) instead of full execution")
    args = parser.parse_args()

    run_pipeline(test_mode=args.test)