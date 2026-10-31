"""
data_audit.py

STEP 2 - Data Audit / EDA for the Amazon ML Challenge 2026
Business Entity Resolution project.

PURPOSE
-------
This script performs a READ-ONLY audit of the provided dataset. It does not
modify, clean, normalize, impute, or filter any input file, and it does not
generate features, candidate pairs, or model artifacts. Its sole purpose is
to produce an evidence-based report (data_audit_report.txt) that downstream
steps (preprocessing, blocking, feature engineering, training) can rely on.

HARD CONSTRAINTS RESPECTED
---------------------------
- All TSVs are read with sep="\t".
- country is treated as an open-set string field; no hard-coded country list.
- No external data, APIs, or lookups are used.
- No rows are dropped, no strings are normalized, no values are imputed.

MEMORY DESIGN
--------------
The dataset is ~1GB on disk. Reading TSVs with dtype=str inflates size in
RAM (Python object overhead per string), so holding all 7 files as full
DataFrames simultaneously could plausibly require several GB of RAM. To
keep peak memory low while preserving every audit requirement, this script:

  1. Processes each source file SEQUENTIALLY. Only one full source
     DataFrame is alive in memory at any time (two, briefly, only when a
     train/test pair of the SAME source number is being compared).
  2. After each file is audited, only small derived artifacts are kept:
     entity_id sets (for referential-integrity / prefix checks), row
     counts, unique-ID counts, country value_counts, missing-value counts,
     and string-length summary stats.
  3. The full DataFrame is deleted (del) and garbage-collected immediately
     after its per-file statistics are extracted.
  4. The ground-truth file is small (Source-1 cardinality, 2 columns) and
     is kept fully in memory for the whole run - this is negligible.
  5. For the human-readable ground-truth sample, instead of keeping full
     source1/2/3 DataFrames alive, we make one additional targeted pass per
     source file that reads only the rows whose entity_id is in a small
     pre-selected sample set, using chunked reading so even this pass never
     materializes the full file at once.

Run from the AmazonML_2026 project root:
    python code/business_entity_resolution/src/data_audit.py
"""

import os
import sys
import gc
import random
from collections import Counter

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASET_TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
DATASET_TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
AUDIT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "code", "business_entity_resolution")
REPORT_PATH = os.path.join(AUDIT_OUTPUT_DIR, "data_audit_report.txt")

SOURCE_FILES = {
    "train_source1": os.path.join(DATASET_TRAIN_DIR, "train_source1.tsv"),
    "train_source2": os.path.join(DATASET_TRAIN_DIR, "train_source2.tsv"),
    "train_source3": os.path.join(DATASET_TRAIN_DIR, "train_source3.tsv"),
    "test_source1": os.path.join(DATASET_TEST_DIR, "test_source1.tsv"),
    "test_source2": os.path.join(DATASET_TEST_DIR, "test_source2.tsv"),
    "test_source3": os.path.join(DATASET_TEST_DIR, "test_source3.tsv"),
}
GROUND_TRUTH_FILE = os.path.join(DATASET_TRAIN_DIR, "train_ground_truth.tsv")

SOURCE_EXPECTED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_EXPECTED_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

PREFIX_BY_SOURCE = {
    "source1": "S1-",
    "source2": "S2-",
    "source3": "S3-",
}

RANDOM_SEED = 42
N_SAMPLE_ENTITIES = 10
CHUNK_SIZE = 200000


class ReportWriter:
    def __init__(self, path):
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")

    def line(self, text=""):
        self._fh.write(str(text) + "\n")

    def header(self, text, level=1):
        bar = "=" * 80 if level == 1 else "-" * 80
        self.line()
        self.line(bar)
        self.line(text)
        self.line(bar)

    def subheader(self, text):
        self.line()
        self.line(f"--- {text} ---")

    def close(self):
        self._fh.close()


def fmt_pct(numerator, denominator):
    if denominator == 0:
        return "N/A (denominator=0)"
    return f"{100.0 * numerator / denominator:.2f}%"


def infer_source_key(filename_key):
    for src in ("source1", "source2", "source3"):
        if src in filename_key:
            return src
    return None


def load_tsv_safely(path, expected_columns, label, rw):
    if not os.path.exists(path):
        msg = f"ERROR: Required file not found: {path}"
        rw.line(msg)
        raise FileNotFoundError(msg)

    try:
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True)
    except Exception as exc:
        msg = f"ERROR: Failed to read {path} as TSV. Exception: {exc}"
        rw.line(msg)
        raise

    rw.subheader(f"File: {label}  ({path})")
    rw.line(f"Rows: {len(df)}")
    rw.line(f"Columns: {len(df.columns)}")
    rw.line(f"Column names (exact): {list(df.columns)}")
    rw.line("Dtypes:")
    for col, dtype in df.dtypes.items():
        rw.line(f"    {col}: {dtype}")
    mem_bytes = df.memory_usage(deep=True).sum()
    rw.line(f"Approx memory usage (deep): {mem_bytes / (1024 ** 2):.2f} MB")

    missing_expected = [c for c in expected_columns if c not in df.columns]
    extra_columns = [c for c in df.columns if c not in expected_columns]
    if missing_expected or extra_columns:
        rw.line("SCHEMA DISCREPANCY DETECTED:")
        if missing_expected:
            rw.line(f"    Missing expected columns: {missing_expected}")
        if extra_columns:
            rw.line(f"    Unexpected extra columns: {extra_columns}")
    else:
        rw.line("Schema check: OK (all expected columns present, no extras).")

    return df


def report_missing_values(df, label, rw):
    rw.subheader(f"Missing values: {label}")
    n_rows = len(df)
    missing_counts = {}
    if n_rows == 0:
        rw.line("No rows to check.")
        return missing_counts
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        missing_counts[col] = n_missing
        rw.line(f"    {col}: {n_missing} missing ({fmt_pct(n_missing, n_rows)})")
    return missing_counts


def report_id_uniqueness(df, label, rw, id_col="entity_id"):
    rw.subheader(f"{id_col} uniqueness: {label}")
    if id_col not in df.columns:
        rw.line(f"    Column '{id_col}' not present; skipping uniqueness check.")
        return None, None
    ids = df[id_col]
    n_total = len(ids)
    n_unique = ids.nunique(dropna=False)
    n_duplicated_rows = n_total - n_unique
    rw.line(f"    Total rows: {n_total}")
    rw.line(f"    Unique {id_col} values: {n_unique}")
    rw.line(f"    Duplicate rows (excess beyond first occurrence): {n_duplicated_rows}")
    if n_duplicated_rows > 0:
        dup_mask = ids.duplicated(keep=False)
        dup_examples = ids[dup_mask].dropna().unique()[:10]
        rw.line(f"    Example duplicate {id_col} values (up to 10): {list(dup_examples)}")
    else:
        rw.line(f"    No duplicate {id_col} values found.")
    return n_total, n_unique


def report_country_distribution(df, label, rw, country_col="country"):
    rw.subheader(f"Country distribution: {label}")
    if country_col not in df.columns:
        rw.line(f"    Column '{country_col}' not present; skipping.")
        return set()
    n_rows = len(df)
    n_missing = int(df[country_col].isna().sum())
    value_counts = df[country_col].value_counts(dropna=True)
    rw.line(f"    Total rows: {n_rows}, missing country: {n_missing} ({fmt_pct(n_missing, n_rows)})")
    rw.line(f"    Number of distinct country values observed: {value_counts.shape[0]}")
    for value, count in value_counts.items():
        rw.line(f"    '{value}': {count} ({fmt_pct(count, n_rows)})")
    return set(value_counts.index)


def report_prefix_validity(df, label, expected_prefix, rw, id_col="entity_id"):
    rw.subheader(f"Prefix validation: {label} (expected prefix '{expected_prefix}')")
    if id_col not in df.columns:
        rw.line(f"    Column '{id_col}' not present; skipping.")
        return None
    ids = df[id_col].dropna()
    n_total = len(ids)
    valid_mask = ids.str.startswith(expected_prefix)
    n_valid = int(valid_mask.sum())
    n_invalid = n_total - n_valid
    rw.line(f"    Total non-null IDs checked: {n_total}")
    rw.line(f"    Valid prefix count: {n_valid} ({fmt_pct(n_valid, n_total)})")
    rw.line(f"    Invalid prefix count: {n_invalid} ({fmt_pct(n_invalid, n_total)})")
    validity_rate = (n_valid / n_total) if n_total else float("nan")
    if n_invalid > 0:
        invalid_examples = ids[~valid_mask].unique()[:10]
        rw.line(f"    Example invalid-prefix IDs (up to 10): {list(invalid_examples)}")
        prefix_counter = Counter(
            (val[: val.index("-") + 1] if "-" in val else val) for val in ids[~valid_mask]
        )
        rw.line(f"    Observed prefixes among invalid IDs: {dict(prefix_counter)}")
    return validity_rate


def report_string_characteristics(df, label, rw, col):
    rw.subheader(f"String characteristics: {label}.{col}")
    if col not in df.columns:
        rw.line(f"    Column '{col}' not present; skipping.")
        return
    series = df[col]
    non_null = series.dropna()
    n_rows = len(series)
    n_non_null = len(non_null)

    if n_non_null == 0:
        rw.line("    No non-null values to analyze.")
        return

    lengths = non_null.str.len()
    stripped = non_null.str.strip()
    n_blank = int((stripped == "").sum())

    rw.line(f"    Non-null values analyzed: {n_non_null} / {n_rows}")
    rw.line(f"    Blank/whitespace-only strings (among non-null): {n_blank} ({fmt_pct(n_blank, n_non_null)})")
    rw.line(f"    Length min: {lengths.min()}")
    rw.line(f"    Length max: {lengths.max()}")
    rw.line(f"    Length mean: {lengths.mean():.2f}")
    rw.line(f"    Length median: {lengths.median():.2f}")
    rw.line(f"    Length std: {lengths.std():.2f}")

    shortest_idx = lengths.nsmallest(3).index
    longest_idx = lengths.nlargest(3).index
    rw.line("    Shortest examples (up to 3):")
    for idx in shortest_idx:
        rw.line(f"        len={lengths.loc[idx]} | value={non_null.loc[idx]!r}")
    rw.line("    Longest examples (up to 3):")
    for idx in longest_idx:
        rw.line(f"        len={lengths.loc[idx]} | value={non_null.loc[idx]!r}")


def extract_entity_id_set(df, id_col="entity_id"):
    if id_col not in df.columns:
        return set()
    return set(df[id_col].dropna().tolist())


def parse_matched_ids(raw_value):
    if pd.isna(raw_value):
        return []
    if raw_value == "":
        return []
    tokens = raw_value.split(",")
    return [t for t in tokens if t != ""]


def analyze_ground_truth(gt_df, rw):
    rw.header("GROUND TRUTH ANALYSIS (train_ground_truth.tsv)")

    if "matched_entity_ids" not in gt_df.columns or "source1_entity_id" not in gt_df.columns:
        rw.line("ERROR: Required ground-truth columns missing; cannot analyze matches.")
        return None, None, None

    gt_row_count, gt_unique_s1_count = report_id_uniqueness(
        gt_df, "train_ground_truth", rw, id_col="source1_entity_id"
    )

    parsed_lists = gt_df["matched_entity_ids"].apply(parse_matched_ids)
    match_counts = parsed_lists.apply(len)

    total_s1_rows = len(gt_df)
    n_zero = int((match_counts == 0).sum())
    n_one = int((match_counts == 1).sum())
    n_multi = int((match_counts > 1).sum())
    max_matches = int(match_counts.max()) if total_s1_rows > 0 else 0

    rw.subheader("Match-count summary per Source 1 entity (ground-truth row basis)")
    rw.line(f"    Total ground-truth rows (source1_entity_id entries): {total_s1_rows}")
    rw.line(f"    Zero-match (singleton) count: {n_zero} ({fmt_pct(n_zero, total_s1_rows)})")
    rw.line(f"    Exactly-one-match count: {n_one} ({fmt_pct(n_one, total_s1_rows)})")
    rw.line(f"    Multi-match (>1) count: {n_multi} ({fmt_pct(n_multi, total_s1_rows)})")
    rw.line(f"    Maximum matches for a single Source 1 entity: {max_matches}")

    rw.subheader("Full distribution of match counts (num_matches: num_source1_entities)")
    dist = match_counts.value_counts().sort_index()
    for n_matches, n_entities in dist.items():
        rw.line(f"    {n_matches} matches -> {n_entities} entities ({fmt_pct(n_entities, total_s1_rows)})")

    rw.subheader("Matched-ID prefix breakdown (across all matches in all rows)")
    all_matched_ids = [mid for sublist in parsed_lists for mid in sublist]
    n_total_matched = len(all_matched_ids)
    n_s2 = sum(1 for mid in all_matched_ids if mid.startswith("S2-"))
    n_s3 = sum(1 for mid in all_matched_ids if mid.startswith("S3-"))
    n_other = n_total_matched - n_s2 - n_s3
    rw.line(f"    Total matched ID occurrences: {n_total_matched}")
    rw.line(f"    S2- prefixed: {n_s2} ({fmt_pct(n_s2, n_total_matched)})")
    rw.line(f"    S3- prefixed: {n_s3} ({fmt_pct(n_s3, n_total_matched)})")
    rw.line(f"    Unexpected/invalid prefixed: {n_other} ({fmt_pct(n_other, n_total_matched)})")
    if n_other > 0:
        other_examples = [mid for mid in all_matched_ids if not (mid.startswith("S2-") or mid.startswith("S3-"))][:10]
        rw.line(f"    Example invalid matched IDs (up to 10): {other_examples}")

    rw.subheader("Within-row duplicate matched IDs")
    dup_row_count = 0
    dup_examples = []
    for s1_id, id_list in zip(gt_df["source1_entity_id"], parsed_lists):
        if len(id_list) != len(set(id_list)):
            dup_row_count += 1
            if len(dup_examples) < 10:
                dup_examples.append((s1_id, id_list))
    rw.line(f"    Rows with duplicate matched IDs within the same row: {dup_row_count} ({fmt_pct(dup_row_count, total_s1_rows)})")
    if dup_examples:
        rw.line("    Examples (source1_entity_id, matched list) up to 10:")
        for s1_id, id_list in dup_examples:
            rw.line(f"        {s1_id}: {id_list}")

    gt_stats = {
        "row_count": gt_row_count,
        "unique_source1_entity_id_count": gt_unique_s1_count,
        "duplicate_source1_entity_id_count": (gt_row_count - gt_unique_s1_count) if gt_row_count is not None else None,
    }

    return parsed_lists, match_counts, gt_stats


def check_referential_integrity(gt_df, parsed_lists, s1_id_set, s2_id_set, s3_id_set, rw):
    rw.header("REFERENTIAL INTEGRITY CHECKS (ground truth vs. source files)")

    rw.subheader("Step 10: source1_entity_id existence in train_source1.tsv")
    gt_s1_ids = gt_df["source1_entity_id"].dropna()
    n_gt_s1 = len(gt_s1_ids)
    missing_mask = ~gt_s1_ids.isin(s1_id_set)
    missing_s1 = gt_s1_ids[missing_mask]
    rw.line(f"    Ground-truth source1_entity_id rows checked: {n_gt_s1}")
    rw.line(f"    Missing from train_source1.tsv: {len(missing_s1)} ({fmt_pct(len(missing_s1), n_gt_s1)})")
    if len(missing_s1) > 0:
        rw.line(f"    Example missing source1_entity_id values (up to 10): {list(missing_s1.unique()[:10])}")

    rw.subheader("Step 11: matched entity IDs existence in train_source2/3.tsv")
    all_matched_ids = [mid for sublist in parsed_lists for mid in sublist]
    n_total_matched = len(all_matched_ids)

    n_found = 0
    n_not_found = 0
    not_found_examples = []
    for mid in all_matched_ids:
        if mid.startswith("S2-"):
            found = mid in s2_id_set
        elif mid.startswith("S3-"):
            found = mid in s3_id_set
        else:
            found = False
        if found:
            n_found += 1
        else:
            n_not_found += 1
            if len(not_found_examples) < 10:
                not_found_examples.append(mid)

    rw.line(f"    Total matched ID references checked: {n_total_matched}")
    rw.line(f"    Found in corresponding source file: {n_found} ({fmt_pct(n_found, n_total_matched)})")
    rw.line(f"    NOT found / unresolved: {n_not_found} ({fmt_pct(n_not_found, n_total_matched)})")
    if not_found_examples:
        rw.line(f"    Example unresolved matched IDs (up to 10): {not_found_examples}")


def select_sample_source1_ids(gt_df, parsed_lists, match_counts):
    rng = random.Random(RANDOM_SEED)

    zero_idx = list(match_counts[match_counts == 0].index)
    one_idx = list(match_counts[match_counts == 1].index)
    multi_idx = list(match_counts[match_counts > 1].index)

    selected_idx = []
    for bucket in (zero_idx, one_idx, multi_idx):
        if bucket:
            rng.shuffle(bucket)
            take = min(3, len(bucket))
            selected_idx.extend(bucket[:take])

    remaining_slots = N_SAMPLE_ENTITIES - len(selected_idx)
    if remaining_slots > 0:
        pool = [i for i in gt_df.index if i not in selected_idx]
        rng.shuffle(pool)
        selected_idx.extend(pool[:remaining_slots])
    selected_idx = selected_idx[:N_SAMPLE_ENTITIES]
    return selected_idx


def targeted_row_lookup(path, wanted_ids, rw, label):
    found = {}
    if not wanted_ids:
        return found
    remaining = set(wanted_ids)
    try:
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True, chunksize=CHUNK_SIZE):
            if "entity_id" not in chunk.columns:
                break
            hit_mask = chunk["entity_id"].isin(remaining)
            if hit_mask.any():
                hits = chunk.loc[hit_mask]
                for _, row in hits.iterrows():
                    found[row["entity_id"]] = {
                        "business_name": row.get("business_name", "N/A"),
                        "business_address": row.get("business_address", "N/A"),
                        "country": row.get("country", "N/A"),
                    }
                remaining -= set(hits["entity_id"].tolist())
            if not remaining:
                break
    except Exception as exc:
        rw.line(f"    WARNING: targeted lookup failed for {label} ({path}): {exc}")
    return found


def sample_ground_truth_relationships(gt_df, parsed_lists, match_counts, rw, csv_out_path):
    rw.header("SAMPLE GROUND-TRUTH RELATIONSHIPS (human-readable)")

    selected_idx = select_sample_source1_ids(gt_df, parsed_lists, match_counts)

    wanted_s1_ids = set(gt_df.loc[selected_idx, "source1_entity_id"].tolist())
    wanted_s2_ids = set()
    wanted_s3_ids = set()
    for i in selected_idx:
        for mid in parsed_lists.loc[i]:
            if mid.startswith("S2-"):
                wanted_s2_ids.add(mid)
            elif mid.startswith("S3-"):
                wanted_s3_ids.add(mid)

    s1_info = targeted_row_lookup(SOURCE_FILES["train_source1"], wanted_s1_ids, rw, "train_source1")
    s2_info = targeted_row_lookup(SOURCE_FILES["train_source2"], wanted_s2_ids, rw, "train_source2")
    s3_info = targeted_row_lookup(SOURCE_FILES["train_source3"], wanted_s3_ids, rw, "train_source3")

    csv_rows = []

    for i in selected_idx:
        s1_id = gt_df.loc[i, "source1_entity_id"]
        matched_ids = parsed_lists.loc[i]

        rw.subheader(f"Source 1 entity: {s1_id}  (matches: {len(matched_ids)})")
        info = s1_info.get(s1_id)
        if info is not None:
            name, addr, country = info["business_name"], info["business_address"], info["country"]
            rw.line(f"    business_name: {name!r}")
            rw.line(f"    business_address: {addr!r}")
            rw.line(f"    country: {country!r}")
        else:
            name, addr, country = "N/A (not found)", "N/A", "N/A"
            rw.line("    WARNING: source1_entity_id not found in train_source1.tsv")

        if not matched_ids:
            rw.line("    Matched records: NONE (singleton)")
        else:
            for mid in matched_ids:
                lookup = s2_info if mid.startswith("S2-") else (s3_info if mid.startswith("S3-") else {})
                minfo = lookup.get(mid)
                if minfo is not None:
                    m_name, m_addr = minfo["business_name"], minfo["business_address"]
                    rw.line(f"    Matched -> {mid}: name={m_name!r}, address={m_addr!r}")
                    csv_rows.append({
                        "source1_entity_id": s1_id,
                        "source1_business_name": name,
                        "source1_business_address": addr,
                        "source1_country": country,
                        "matched_entity_id": mid,
                        "matched_business_name": m_name,
                        "matched_business_address": m_addr,
                    })
                else:
                    rw.line(f"    Matched -> {mid}: WARNING not found in source files")
                    csv_rows.append({
                        "source1_entity_id": s1_id,
                        "source1_business_name": name,
                        "source1_business_address": addr,
                        "source1_country": country,
                        "matched_entity_id": mid,
                        "matched_business_name": "NOT_FOUND",
                        "matched_business_address": "NOT_FOUND",
                    })
        if not matched_ids:
            csv_rows.append({
                "source1_entity_id": s1_id,
                "source1_business_name": name,
                "source1_business_address": addr,
                "source1_country": country,
                "matched_entity_id": "",
                "matched_business_name": "",
                "matched_business_address": "",
            })

    if csv_rows:
        pd.DataFrame(csv_rows).to_csv(csv_out_path, index=False)
        rw.line()
        rw.line(f"[Saved sample relationships CSV to: {csv_out_path}]")


def compare_train_test_for_source(src_num, rw):
    train_key = f"train_source{src_num}"
    test_key = f"test_source{src_num}"
    train_path = SOURCE_FILES[train_key]
    test_path = SOURCE_FILES[test_key]

    train_df = pd.read_csv(train_path, sep="\t", dtype=str, keep_default_na=True)
    test_df = pd.read_csv(test_path, sep="\t", dtype=str, keep_default_na=True)

    rw.subheader(f"Source {src_num}: train vs test")
    rw.line(f"    train rows: {len(train_df)}   test rows: {len(test_df)}")
    rw.line(f"    train columns: {list(train_df.columns)}")
    rw.line(f"    test columns: {list(test_df.columns)}")
    if list(train_df.columns) != list(test_df.columns):
        rw.line("    SCHEMA DISCREPANCY: train and test columns differ for this source.")
    else:
        rw.line("    Schema match: OK")

    for col in train_df.columns:
        if col in test_df.columns:
            tr_missing = int(train_df[col].isna().sum())
            te_missing = int(test_df[col].isna().sum())
            rw.line(
                f"    Missing in '{col}': train={tr_missing} ({fmt_pct(tr_missing, len(train_df))}), "
                f"test={te_missing} ({fmt_pct(te_missing, len(test_df))})"
            )

    if "entity_id" in train_df.columns and "entity_id" in test_df.columns:
        expected_prefix = PREFIX_BY_SOURCE[f"source{src_num}"]
        tr_valid = train_df["entity_id"].dropna().str.startswith(expected_prefix).mean() if len(train_df) else float("nan")
        te_valid = test_df["entity_id"].dropna().str.startswith(expected_prefix).mean() if len(test_df) else float("nan")
        rw.line(f"    Prefix validity rate (expected '{expected_prefix}'): train={tr_valid:.2%}, test={te_valid:.2%}")

    if "country" in train_df.columns and "country" in test_df.columns:
        train_countries = set(train_df["country"].dropna().unique())
        test_countries = set(test_df["country"].dropna().unique())
        only_in_test = test_countries - train_countries
        only_in_train = train_countries - test_countries
        rw.line(f"    Countries in train: {sorted(train_countries)}")
        rw.line(f"    Countries in test: {sorted(test_countries)}")
        rw.line(f"    Countries in TEST but NOT in train (informational, not an error): {sorted(only_in_test)}")
        rw.line(f"    Countries in train but NOT in test: {sorted(only_in_train)}")

    del train_df, test_df
    gc.collect()


def main():
    os.makedirs(AUDIT_OUTPUT_DIR, exist_ok=True)
    rw = ReportWriter(REPORT_PATH)

    rw.header("AMAZON ML CHALLENGE 2026 - BUSINESS ENTITY RESOLUTION")
    rw.line("DATA AUDIT REPORT (read-only; no data was modified)")
    rw.line(f"Project root resolved as: {PROJECT_ROOT}")
    rw.line("Memory strategy: files are processed sequentially; only small derived")
    rw.line("artifacts (ID sets, row counts, unique counts, summary stats) are")
    rw.line("retained across files.")

    id_sets = {}
    row_counts = {}
    unique_id_counts = {}

    try:
        rw.header("STEP 1-3: FILE LOADING, SHAPE, SCHEMA CHECK")
        rw.header("STEP 4: MISSING VALUE ANALYSIS", level=2)
        rw.header("STEP 5: ENTITY_ID UNIQUENESS PER SOURCE FILE", level=2)
        rw.header("STEP 6: COUNTRY DISTRIBUTION PER SOURCE (OPEN-SET)", level=2)
        rw.header("STEP 7: SOURCE PREFIX VALIDATION", level=2)
        rw.header("STEP 13: STRING LENGTH CHARACTERISTICS", level=2)
        rw.line("(Steps 1-7 and 13 are interleaved per file below to avoid re-loading each file multiple times.)")

        for key, path in SOURCE_FILES.items():
            rw.header(f"PROCESSING FILE: {key}")
            df = load_tsv_safely(path, SOURCE_EXPECTED_COLUMNS, key, rw)

            row_counts[key] = len(df)

            report_missing_values(df, key, rw)
            n_total, n_unique = report_id_uniqueness(df, key, rw, id_col="entity_id")
            unique_id_counts[key] = n_unique

            report_country_distribution(df, key, rw)

            src = infer_source_key(key)
            if src is not None:
                report_prefix_validity(df, key, PREFIX_BY_SOURCE[src], rw)

            report_string_characteristics(df, key, rw, "business_name")
            report_string_characteristics(df, key, rw, "business_address")

            id_sets[key] = extract_entity_id_set(df)

            del df
            gc.collect()

        rw.header("PROCESSING FILE: train_ground_truth")
        gt_df = load_tsv_safely(GROUND_TRUTH_FILE, GT_EXPECTED_COLUMNS, "train_ground_truth", rw)
        gt_total_rows = len(gt_df)
        report_missing_values(gt_df, "train_ground_truth", rw)

    except (FileNotFoundError, Exception) as exc:
        rw.line(f"\nFATAL: Audit aborted due to load error: {exc}")
        rw.close()
        print(f"Audit aborted. See {REPORT_PATH} for details.", file=sys.stderr)
        sys.exit(1)

    parsed_lists, match_counts, gt_stats = analyze_ground_truth(gt_df, rw)
    if parsed_lists is None:
        parsed_lists, match_counts = pd.Series(dtype=object), pd.Series(dtype=int)
        gt_stats = {
            "row_count": gt_total_rows,
            "unique_source1_entity_id_count": None,
            "duplicate_source1_entity_id_count": None,
        }

    check_referential_integrity(
        gt_df, parsed_lists,
        id_sets.get("train_source1", set()),
        id_sets.get("train_source2", set()),
        id_sets.get("train_source3", set()),
        rw,
    )

    sample_csv_path = os.path.join(AUDIT_OUTPUT_DIR, "sample_ground_truth_relationships.csv")
    if len(gt_df) > 0:
        sample_ground_truth_relationships(gt_df, parsed_lists, match_counts, rw, sample_csv_path)

    rw.header("STEP 14: TRAIN VS TEST HIGH-LEVEL COMPARISON")
    for src_num in ("1", "2", "3"):
        compare_train_test_for_source(src_num, rw)

    rw.header("DATA AUDIT SUMMARY")
    rw.line("Key findings relevant to downstream decisions:")
    rw.line("")

    n_zero = int((match_counts == 0).sum()) if len(match_counts) else 0
    n_one = int((match_counts == 1).sum()) if len(match_counts) else 0
    n_multi = int((match_counts > 1).sum()) if len(match_counts) else 0
    max_matches = int(match_counts.max()) if len(match_counts) else 0

    rw.subheader("Row counts (raw number of rows read from each file)")
    for key in SOURCE_FILES:
        rw.line(f"    {key}: {row_counts.get(key)} rows")
    rw.line(f"    train_ground_truth: {gt_stats.get('row_count')} rows")

    rw.subheader("Unique entity_id / source1_entity_id counts (distinct IDs, not row counts)")
    for key in SOURCE_FILES:
        rw.line(f"    {key}: {unique_id_counts.get(key)} unique entity_id values")
    rw.line(
        f"    train_ground_truth: {gt_stats.get('unique_source1_entity_id_count')} unique "
        f"source1_entity_id values (duplicates: {gt_stats.get('duplicate_source1_entity_id_count')})"
    )
    rw.line("")
    rw.line("    NOTE: 'row count' is the total number of lines read from a file (including any")
    rw.line("    duplicate IDs); 'unique ID count' is the number of distinct entity_id /")
    rw.line("    source1_entity_id values. If a file has zero duplicates these two numbers match;")
    rw.line("    any difference is a duplication signal and is reported explicitly above in the")
    rw.line("    per-file uniqueness sections and in the ground-truth uniqueness check.")
    rw.line("")

    rw.line(f"1. Ground truth match distribution (row basis): zero={n_zero} ({fmt_pct(n_zero, gt_stats.get('row_count') or 0)}), "
            f"one={n_one} ({fmt_pct(n_one, gt_stats.get('row_count') or 0)}), multi={n_multi} ({fmt_pct(n_multi, gt_stats.get('row_count') or 0)}), "
            f"max_matches_for_one_entity={max_matches}.")
    rw.line("   -> This shapes class balance for pair-classification training and confirms singleton")
    rw.line("      handling (empty matched_entity_ids) must be a first-class prediction outcome, not an edge case.")
    rw.line("")
    rw.line("2. source1_entity_id uniqueness in train_ground_truth.tsv is validated explicitly above")
    rw.line("   (see GROUND TRUTH ANALYSIS section). Since Source 1 is the deduplicated reference source,")
    rw.line("   any duplicate source1_entity_id rows in ground truth would indicate conflicting or")
    rw.line("   redundant labels that must be resolved explicitly before building training pairs.")
    rw.line("")
    rw.line("3. Review the SCHEMA DISCREPANCY and prefix-validation sections above before assuming clean IDs;")
    rw.line("   any reported discrepancy must be resolved explicitly in preprocessing, not silently patched here.")
    rw.line("")
    rw.line("4. Review the COUNTRY DISTRIBUTION and TRAIN VS TEST sections above for the observed open-set")
    rw.line("   country values, including any countries present in test but absent from train. Country-specific")
    rw.line("   address patterns (if used in feature engineering) must generalize to unseen country labels.")
    rw.line("")
    rw.line("5. Review REFERENTIAL INTEGRITY results above: any unresolved matched IDs or missing")
    rw.line("   source1_entity_id values indicate ground-truth rows that cannot be used as-is for supervised")
    rw.line("   pair labels and must be handled explicitly (not silently dropped) in a later step.")
    rw.line("")
    rw.line("6. Review MISSING VALUE percentages per column above to decide which fields are reliable enough")
    rw.line("   to drive blocking keys (e.g., business_name) versus fields needing fallback logic (e.g., address).")
    rw.line("")
    rw.line("7. Review STRING LENGTH statistics above: extreme lengths or many blank strings affect the choice")
    rw.line("   of blocking granularity (e.g., token/n-gram blocking) and similarity feature design.")
    rw.line("")
    rw.line("8. Memory note: this audit processed each source file sequentially, retaining only ID sets, row")
    rw.line("   counts, and summary statistics across files, to keep peak RAM well below the full")
    rw.line("   ~1GB*expansion footprint that loading all 7 files simultaneously would require.")
    rw.line("")
    rw.line("This report is descriptive only. No preprocessing, blocking, feature engineering, or modeling")
    rw.line("decisions have been applied to the data by this script.")

    rw.close()
    print(f"Data audit complete. Report written to: {REPORT_PATH}")
    if len(gt_df) > 0:
        print(f"Sample relationships CSV written to: {sample_csv_path}")


if __name__ == "__main__":
    main()