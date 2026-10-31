"""
preprocessing.py

STEP 3 - Preprocessing for the Amazon ML Challenge 2026
Business Entity Resolution project.

SCOPE (STRICT)
--------------
This module performs ONLY conservative, deterministic, rule-based text
normalization on the raw source columns. It does NOT perform candidate
generation, blocking, fuzzy matching, similarity scoring, ML modeling,
threshold optimization, prediction, or submission generation. Those belong
to later pipeline stages (blocking.py, features.py, train.py, predict.py).

DESIGN PRINCIPLES (grounded in the actual data_audit_report.txt findings)
---------------------------------------------------------------------------
1. Raw columns (entity_id, business_name, business_address, country) are
   NEVER overwritten. They are always retained unchanged in the output.
2. A conservative PRIMARY normalized representation is produced for each
   text field. It preserves Unicode scripts, digits, and token content -
   it only fixes case, whitespace, and Unicode-equivalence noise.
   The audit's own ground-truth sample shows true matches written in
   Devanagari and Tamil/Telugu scripts, so destructive ASCII
   transliteration would destroy real matching evidence and is never
   performed.
3. A small number of SECONDARY representations are added only where the
   audit shows a concrete, recurring noise pattern that a primary
   conservative normalization cannot address on its own:
     - punctuation/separator noise
     - business-name / address token reordering, e.g.
       "Orthopedic Associates Inc" vs. "Orthopedic Inc Associates"
       (audit sample, ground-truth match)
   Secondary representations never replace the primary representation.
4. Missing values remain missing (None / pandas NA / float NaN / NaT).
   The audit confirms real missing business_address values (3.36% in
   train_source2, 3.33% in train_source3, similar rates in
   test_source2/3), including in confirmed ground-truth matches. These
   are never coerced into the literal strings "nan", "none", "null",
   "<NA>", or "missing" at ANY stage of the pipeline - primary
   normalization AND every secondary representation.
5. country is treated as a fully open string set. The audit shows France
   appears only in test_source*.tsv (not in any train file); no country
   list, mapping, or fallback logic is hard-coded anywhere in this module.
6. No ground-truth or test-set-derived learning of any kind occurs. All
   functions are pure, deterministic, rule-based transformations that use
   only the value passed to them.
7. Files are processed one at a time, in chunks, with each chunk written
   to disk immediately and released. No full source file and no
   concatenated "all-chunks" DataFrame is ever held in memory.

MISSING-VALUE HANDLING (HARDENED, DEFENSE-IN-DEPTH)
-----------------------------------------------------
Every function that receives a possibly-missing value - whether called
directly or via Series.map() - applies TWO independent layers of
protection before performing any string-only operation (iteration,
.split(), .strip(), etc.):

  Layer 1: _is_missing(value) - explicitly recognizes None, pd.NA,
           pd.NaT, and any value for which pd.isna() returns a scalar
           True (this covers float('nan') in all its forms). If
           pd.isna() returns something that is not a plain bool (e.g.
           an array, because `value` was list-like), this layer treats
           the value as "not a recognized scalar missing value" and
           defers to Layer 2.
  Layer 2: isinstance(value, str) - an explicit type gate. If a value
           somehow reaches a normalization function without being
           resolved as missing by Layer 1 (for example, an unexpected
           type coming from a pandas version/dtype quirk, a list, a
           dict, or any other non-string object), Layer 2 guarantees
           the function still returns None instead of attempting a
           string-only operation on a non-string object. This makes it
           structurally impossible for `for ch in normalized_value` or
           `normalized_value.split(...)` to ever execute on a non-string
           value, regardless of pandas version or dtype behavior.

This two-layer approach is applied uniformly to normalize_business_name(),
normalize_business_address(), normalize_country(), to_alnum_representation(),
to_token_list(), and to_token_sorted_representation() - there is exactly
ONE missing-value policy in this module, not two incompatible ones.

PATH RESOLUTION (VERIFIED)
---------------------------
This file lives at:
    Amazon_ML_2026/Code/business_entity_resolution/src/preprocessing.py

From that location, exactly THREE ".." levels reach the project root:
    src -> business_entity_resolution -> Code -> Amazon_ML_2026 (root)

Run from the Amazon_ML_2026 project root:
    python Code/business_entity_resolution/src/preprocessing.py --demo
    python Code/business_entity_resolution/src/preprocessing.py --file train_source1
"""

from __future__ import annotations

import os
import re
import sys
import argparse
import unicodedata
from typing import Optional, List

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (relative to project root: Amazon_ML_2026/)
# From .../Code/business_entity_resolution/src/preprocessing.py, three levels
# up (src -> business_entity_resolution -> Code) reaches the project root.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASET_TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
DATASET_TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "preprocessed")

SOURCE_FILES = {
    "train_source1": os.path.join(DATASET_TRAIN_DIR, "train_source1.tsv"),
    "train_source2": os.path.join(DATASET_TRAIN_DIR, "train_source2.tsv"),
    "train_source3": os.path.join(DATASET_TRAIN_DIR, "train_source3.tsv"),
    "test_source1": os.path.join(DATASET_TEST_DIR, "test_source1.tsv"),
    "test_source2": os.path.join(DATASET_TEST_DIR, "test_source2.tsv"),
    "test_source3": os.path.join(DATASET_TEST_DIR, "test_source3.tsv"),
}

REQUIRED_RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

DEFAULT_CHUNKSIZE = 200_000

# Output columns, in the order they will be written. Raw columns first
# (unchanged), then normalized/secondary derived columns. Address gets the
# same token-sorted secondary representation as business_name, for
# symmetry and because address component reordering is also observed in
# the audit's ground-truth-confirmed match pairs.
OUTPUT_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
    "business_name_normalized",
    "business_name_alnum",
    "business_name_tokens",
    "business_name_token_sorted",
    "business_address_normalized",
    "business_address_alnum",
    "business_address_tokens",
    "business_address_token_sorted",
    "country_normalized",
]

# ---------------------------------------------------------------------------
# Low-level Unicode / missing-value helpers (shared, deterministic)
# ---------------------------------------------------------------------------

def _is_missing(value) -> bool:
    """
    Layer 1 of the missing-value policy: the single source of truth for
    "is this value missing" across the whole module. Returns True for:
      - Python None
      - pandas NA (pd.NA)
      - pandas NaT (pd.NaT)
      - float NaN (float('nan')), which is what pandas actually stores in
        object/float columns for missing cells in many pandas versions
        and code paths.

    Never returns True for an empty string "" - an empty string is a
    valid (if unusual) observed value, not a missing one.

    Never returns True for list-like/array-like values either: pd.isna()
    on a list or array returns an array of booleans rather than a single
    boolean, so such a result is explicitly treated as "not a recognized
    scalar missing value" here (result: False), and the value is instead
    handled by the isinstance(value, str) Layer 2 gate downstream, which
    safely converts any non-string into None rather than raising.
    """
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        # pd.isna() can raise on some exotic types; fail safe (not missing)
        # rather than crash - Layer 2 (_is_usable_string) is the backstop.
        return False
    if isinstance(result, bool):
        return result
    # pd.isna() returned an array-like (e.g. input was a list/ndarray) -
    # this is not a scalar missing value; let Layer 2 handle it safely.
    return False


def _is_usable_string(value) -> bool:
    """
    Layer 2 of the missing-value policy: an explicit, unconditional type
    gate. Returns True only if `value` is an actual Python str instance.

    This is checked IN ADDITION to _is_missing(), never instead of it.
    Its purpose is to make it structurally impossible for any
    string-only operation (iteration, .split(), .strip(), regex
    matching, etc.) to ever execute against a non-string object -
    including float('nan'), pd.NA, pd.NaT, or any other unexpected type
    that might reach a normalization function through a pandas
    Series.map() call, regardless of pandas version or dtype behavior.
    """
    return isinstance(value, str)


def _nfkc_and_collapse_whitespace(text: str) -> str:
    """
    Apply Unicode NFKC normalization, normalize all Unicode whitespace
    (category Zs, plus tab/newline/CR/vertical tab/form feed) to a single
    ASCII space, strip leading/trailing whitespace, and collapse repeated
    whitespace. This is script-agnostic and non-destructive to letters,
    digits, or any script's characters.

    Precondition: `text` must already be a str. Callers are responsible
    for missing-value / type checks before calling this helper.
    """
    text = unicodedata.normalize("NFKC", text)
    normalized_chars = []
    for ch in text:
        if ch in ("\t", "\n", "\r", "\v", "\f") or unicodedata.category(ch) == "Zs":
            normalized_chars.append(" ")
        else:
            normalized_chars.append(ch)
    text = "".join(normalized_chars)
    text = text.strip()
    text = re.sub(r" {2,}", " ", text)
    return text


# ---------------------------------------------------------------------------
# PRIMARY normalization functions (conservative, information-preserving)
# ---------------------------------------------------------------------------

def normalize_business_name(value: Optional[str]) -> Optional[str]:
    """
    Conservative primary normalization of a business_name value.

    Steps (safe, non-destructive, order matters):
      1. Missing-value passthrough via _is_missing() (Layer 1).
      2. Non-string passthrough via _is_usable_string() (Layer 2) - any
         remaining non-string value is treated as missing rather than
         coerced or iterated over.
      3. Unicode NFKC normalization (fixes compatibility variants, e.g.
         full-width characters, without altering meaningful content).
      4. Unicode whitespace normalization + collapse + strip.
      5. Unicode-aware casefold() (not .lower()) for robust case folding
         across scripts.

    Does NOT: remove punctuation, remove legal suffixes, sort tokens,
    perform fuzzy correction, or use any external dictionary. This keeps
    the representation information-rich for later matching stages.
    """
    if _is_missing(value):
        return None
    if not _is_usable_string(value):
        return None
    text = _nfkc_and_collapse_whitespace(value)
    text = text.casefold()
    return text


def normalize_business_address(value: Optional[str]) -> Optional[str]:
    """
    Conservative primary normalization of a business_address value.

    Applies the same safe Unicode/whitespace/casefold pipeline as
    normalize_business_name, protected by the same two-layer
    missing-value/type check. Digits, postal codes, building/unit
    numbers, and non-Latin scripts are preserved untouched (no
    geocoding, no translation, no inference of missing components).
    This is the field with the highest real missing rate in the dataset
    (up to ~3.4% in Source 2/3), so this path is exercised heavily on
    the real data.
    """
    if _is_missing(value):
        return None
    if not _is_usable_string(value):
        return None
    text = _nfkc_and_collapse_whitespace(value)
    text = text.casefold()
    return text


def normalize_country(value: Optional[str]) -> Optional[str]:
    """
    Conservative, open-set-safe normalization of a country value.

    Only performs: missing-value handling (two-layer check), Unicode
    NFKC normalization, whitespace normalization/trim, and casefold().
    Does NOT map, bucket, infer, translate, or restrict to any fixed set
    of country labels - the test set is known (from the audit) to contain
    'France', which does not appear anywhere in the training data, and
    any genuinely different country string must remain distinguishable
    after normalization.
    """
    if _is_missing(value):
        return None
    if not _is_usable_string(value):
        return None
    text = _nfkc_and_collapse_whitespace(value)
    text = text.casefold()
    return text


# ---------------------------------------------------------------------------
# SECONDARY representations (narrowly justified, clearly separated)
#
# Each function below applies BOTH _is_missing() AND _is_usable_string()
# before any string-only operation. This makes it structurally
# impossible for `for ch in normalized_value` or
# `normalized_value.split(...)` to ever run against a non-string value.
# ---------------------------------------------------------------------------

# Unicode-aware "word" pattern: letters/marks/digits from any script,
# grouped into tokens. This deliberately avoids any ASCII-only regex like
# [A-Za-z0-9] so that Devanagari, Tamil, Telugu, Kannada, accented Latin,
# etc. are preserved as valid token characters instead of being stripped.
_WORD_CHAR_RE = re.compile(r"[^\W\d_]|\d", re.UNICODE)
_TOKEN_SPLIT_RE = re.compile(r"[^\w]+", re.UNICODE)


def to_alnum_representation(normalized_value: Optional[str]) -> Optional[str]:
    """
    Secondary representation: remove punctuation/separator noise while
    keeping letters (any script) and digits, collapsing the result to
    single spaces between remaining tokens.

    Purpose: neutralize stray punctuation/OCR-like noise observed in the
    audit's ground-truth-confirmed match pairs, without discarding the
    primary normalized string, which is kept intact for exact/near-exact
    comparison in later stages.

    Two-layer missing-value/type guard: _is_missing(normalized_value) is
    checked first, THEN _is_usable_string(normalized_value) is checked
    before the `for ch in normalized_value` loop below. Together these
    make it impossible for this loop to ever execute on a float, pd.NA,
    pd.NaT, or any other non-string object - this is the exact fix for
    the reported "TypeError: 'float' object is not iterable" crash.
    """
    if _is_missing(normalized_value):
        return None
    if not _is_usable_string(normalized_value):
        return None
    kept_chars = []
    prev_was_space = False
    for ch in normalized_value:
        if _WORD_CHAR_RE.match(ch):
            kept_chars.append(ch)
            prev_was_space = False
        else:
            if not prev_was_space and kept_chars:
                kept_chars.append(" ")
                prev_was_space = True
    result = "".join(kept_chars).strip()
    result = re.sub(r" {2,}", " ", result)
    return result if result != "" else None


def to_token_list(normalized_value: Optional[str]) -> Optional[str]:
    """
    Secondary representation: whitespace/punctuation-delimited token list,
    encoded as a single space-joined string (kept as a string, not a
    Python list, so it round-trips cleanly through a TSV column).

    Duplicate tokens are preserved intentionally (e.g. repeated words in
    an address are not assumed to be noise). No stopword removal, no
    minimum-length filtering, no stemming.

    Two-layer missing-value/type guard, identical in structure to
    to_alnum_representation(): _is_missing() first, then
    _is_usable_string(), before _TOKEN_SPLIT_RE.split(normalized_value)
    is attempted. Splitting a non-string would otherwise raise
    AttributeError/TypeError.
    """
    if _is_missing(normalized_value):
        return None
    if not _is_usable_string(normalized_value):
        return None
    tokens = [t for t in _TOKEN_SPLIT_RE.split(normalized_value) if t != ""]
    if not tokens:
        return None
    return " ".join(tokens)


def to_token_sorted_representation(token_list_value: Optional[str]) -> Optional[str]:
    """
    Secondary representation ONLY: lexicographically sorted tokens.

    Purpose: expose business-name / address relationships where word
    order differs, matching the audit-confirmed ground-truth pair
    "Orthopedic Associates Inc" (S1) <-> "Orthopedic Inc Associates" (S2).

    This is never used as, or substituted for, business_name_normalized
    or business_address_normalized. Duplicate tokens are preserved
    (sorting does not deduplicate).

    Two-layer missing-value/type guard, identical in structure to the
    other secondary functions: _is_missing() first, then
    _is_usable_string(), before token_list_value.split(" ") is
    attempted.
    """
    if _is_missing(token_list_value):
        return None
    if not _is_usable_string(token_list_value):
        return None
    tokens = token_list_value.split(" ")
    tokens.sort()
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Column-level orchestration for a single chunk/DataFrame
# ---------------------------------------------------------------------------

def add_normalized_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame containing the raw required columns, return a NEW
    DataFrame containing the raw columns UNCHANGED plus the normalized /
    secondary derived columns. The input DataFrame is not mutated in
    place beyond building a working copy, and no raw column is ever
    overwritten.

    Every .map() call below routes through functions that apply the
    two-layer missing-value/type guard internally, so this is safe even
    when business_name, business_address, or country contain a
    substantial fraction of missing values (as business_address
    genuinely does in Source 2/3), regardless of whether pandas
    represents those missing cells as None, pd.NA, or float NaN.
    """
    missing_required = [c for c in REQUIRED_RAW_COLUMNS if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"Input chunk is missing required raw columns: {missing_required}. "
            f"Found columns: {list(df.columns)}"
        )

    out = df.copy()

    out["business_name_normalized"] = out["business_name"].map(normalize_business_name)
    out["business_address_normalized"] = out["business_address"].map(normalize_business_address)
    out["country_normalized"] = out["country"].map(normalize_country)

    out["business_name_alnum"] = out["business_name_normalized"].map(to_alnum_representation)
    out["business_name_tokens"] = out["business_name_normalized"].map(to_token_list)
    out["business_name_token_sorted"] = out["business_name_tokens"].map(to_token_sorted_representation)

    out["business_address_alnum"] = out["business_address_normalized"].map(to_alnum_representation)
    out["business_address_tokens"] = out["business_address_normalized"].map(to_token_list)
    out["business_address_token_sorted"] = out["business_address_tokens"].map(to_token_sorted_representation)

    out = out[OUTPUT_COLUMNS]
    return out


# ---------------------------------------------------------------------------
# Chunked, memory-conscious file processing
# ---------------------------------------------------------------------------

def process_file_in_chunks(
    input_path: str,
    output_path: str,
    chunksize: int = DEFAULT_CHUNKSIZE,
) -> int:
    """
    Stream `input_path` (a TSV with sep="\t") in chunks of `chunksize`
    rows, apply add_normalized_columns() to each chunk, and append the
    result to `output_path` (also tab-separated) incrementally.

    Memory behavior:
      - Only one chunk (raw) and its transformed counterpart are held in
        memory at any time.
      - The output file is opened in write mode for the first chunk (to
        write the header) and append mode thereafter; each chunk is
        flushed to disk and then released (no accumulation).
      - No full-file DataFrame is ever constructed.

    Returns the total number of rows processed.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_rows = 0
    first_chunk = True

    reader = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=True,
        chunksize=chunksize,
    )

    for chunk in reader:
        processed = add_normalized_columns(chunk)

        mode = "w" if first_chunk else "a"
        header = first_chunk
        processed.to_csv(
            output_path,
            sep="\t",
            mode=mode,
            header=header,
            index=False,
            na_rep="",
        )
        total_rows += len(processed)

        del chunk, processed
        first_chunk = False

    return total_rows


# ---------------------------------------------------------------------------
# Lightweight validation
# ---------------------------------------------------------------------------

def validate_normalization() -> List[str]:
    """
    Run a set of lightweight, in-memory checks against representative
    strings (no large file I/O). Returns a list of human-readable failure
    messages; an empty list means all checks passed.

    Critically, this function calls add_normalized_columns() itself -
    the EXACT same function run_demo() calls - on a DataFrame built the
    same way (a Python list containing None among strings, which pandas
    stores as an object-dtype column). This ensures validation exercises
    the real Series.map() path end-to-end, not just the bare functions
    in isolation, so a regression like the reported crash cannot slip
    through with "ALL PASSED" printed right before a real crash.
    """
    failures: List[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    missing_variants = [None, pd.NA, float("nan"), pd.NaT]
    literal_missing_strings = (
        "nan", "none", "null", "missing", "<na>", "<nat>",
        "NaN", "None", "NULL", "NAN", "NONE", "<NA>", "<NaT>",
    )

    # 1. Primary normalization functions must safely handle every missing
    #    variant and never produce a literal missing-like string.
    for missing_value in missing_variants:
        check(normalize_business_name(missing_value) is None, f"normalize_business_name({missing_value!r}) must be None")
        check(normalize_business_address(missing_value) is None, f"normalize_business_address({missing_value!r}) must be None")
        check(normalize_country(missing_value) is None, f"normalize_country({missing_value!r}) must be None")

    # 2. Secondary representation functions, called directly, must ALSO
    #    safely handle every missing variant.
    for missing_value in missing_variants:
        check(to_alnum_representation(missing_value) is None, f"to_alnum_representation({missing_value!r}) must be None")
        check(to_token_list(missing_value) is None, f"to_token_list({missing_value!r}) must be None")
        check(to_token_sorted_representation(missing_value) is None, f"to_token_sorted_representation({missing_value!r}) must be None")

    # 3. No function anywhere may turn a missing value into a literal
    #    missing-like string.
    all_functions_under_test = [
        normalize_business_name, normalize_business_address, normalize_country,
        to_alnum_representation, to_token_list, to_token_sorted_representation,
    ]
    for fn in all_functions_under_test:
        for missing_value in missing_variants:
            result = fn(missing_value)
            check(
                result is None or str(result).lower() not in [s.lower() for s in literal_missing_strings],
                f"{fn.__name__}({missing_value!r}) must not produce a literal missing-like string, got {result!r}",
            )

    # 4. Robustness against unexpected non-string, non-missing types
    #    (defense-in-depth Layer 2: _is_usable_string). These are not
    #    expected in real data but must not crash the pipeline.
    unexpected_types = [42, 3.14, True, ["a", "b"], {"k": "v"}]
    for fn in all_functions_under_test:
        for weird_value in unexpected_types:
            try:
                result = fn(weird_value)
                check(
                    result is None or isinstance(result, str),
                    f"{fn.__name__}({weird_value!r}) must return None or str, got {result!r}",
                )
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{fn.__name__}({weird_value!r}) must not raise, got {exc}")

    # 5. Determinism: same input -> same output, repeated calls.
    sample_name = "  Kestrel  Chicago   Associates  Inc. "
    check(normalize_business_name(sample_name) == normalize_business_name(sample_name), "normalize_business_name must be deterministic")

    # 6. Unicode preservation: non-Latin scripts must survive intact.
    devanagari = "व्हाइट प्रोड्यूसर प्राइवेट लिमिटेड"
    normalized_devanagari = normalize_business_name(devanagari)
    check(normalized_devanagari is not None and "व" in normalized_devanagari, "Devanagari must survive business_name normalization")
    tamil = "தமிழ்நாடு"
    normalized_tamil = normalize_business_address(tamil)
    check(normalized_tamil is not None and "த" in normalized_tamil, "Tamil must survive business_address normalization")

    # 7. Numbers must survive address normalization.
    address_with_numbers = "5101 Disbrow Court, Apt 4B, Lincoln, NE 68502"
    normalized_address = normalize_business_address(address_with_numbers)
    for digit_group in ("5101", "4", "68502"):
        check(normalized_address is not None and digit_group in normalized_address, f"Digits '{digit_group}' must survive address normalization")

    # 8. Countries must not be accidentally collapsed together.
    us_norm, india_norm, france_norm = normalize_country("US"), normalize_country("India"), normalize_country("France")
    check(len({us_norm, india_norm, france_norm}) == 3, "Distinct country values must remain distinguishable")

    # 9. Token sorting must be secondary only (business name).
    name_a, name_b = "Orthopedic Associates Inc", "Orthopedic Inc Associates"
    norm_a, norm_b = normalize_business_name(name_a), normalize_business_name(name_b)
    tokens_a, tokens_b = to_token_list(norm_a), to_token_list(norm_b)
    sorted_a, sorted_b = to_token_sorted_representation(tokens_a), to_token_sorted_representation(tokens_b)
    check(norm_a != norm_b, "Primary normalized names should retain original token order (differ here)")
    check(sorted_a == sorted_b, "Token-sorted secondary representation should expose the reordering relationship")
    check(norm_a != sorted_a, "Primary normalized name must not equal the token-sorted secondary representation")

    # 10. Output schema determinism.
    demo_df = pd.DataFrame({
        "entity_id": ["S1-000001"], "business_name": ["Acme Corp"],
        "business_address": ["1 Main St"], "country": ["US"],
    })
    result_df = add_normalized_columns(demo_df)
    check(list(result_df.columns) == OUTPUT_COLUMNS, "add_normalized_columns must produce a deterministic column schema")
    check(result_df.loc[0, "business_name"] == "Acme Corp", "Raw business_name column must remain unchanged")
    check(result_df.loc[0, "business_address"] == "1 Main St", "Raw business_address column must remain unchanged")
    check(result_df.loc[0, "country"] == "US", "Raw country column must remain unchanged")

    # 11. Required raw columns must be enforced.
    incomplete_df = pd.DataFrame({"entity_id": ["S1-1"], "business_name": ["X"]})
    try:
        add_normalized_columns(incomplete_df)
        failures.append("add_normalized_columns must raise ValueError when required raw columns are missing")
    except ValueError:
        pass

    # 12. THE EXACT REGRESSION TEST, matching run_demo()'s construction
    #     pattern (a Python list with None mixed into strings, exactly
    #     like demo_raw's business_address column): a normal business
    #     name/address, a row with a None address, a row with a pd.NA
    #     address, and a row with a float("nan") address, run through the
    #     FULL add_normalized_columns() pipeline (the exact function
    #     run_demo() calls) - covering business_name_alnum,
    #     business_name_tokens, business_name_token_sorted,
    #     business_address_alnum, business_address_tokens, and
    #     business_address_token_sorted.
    missing_address_df = pd.DataFrame(
        {
            "entity_id": ["S1-100001", "S1-100002", "S1-100003", "S1-100004"],
            "business_name": ["Acme Corp", "Beta LLC", "Gamma Traders", "Delta Inc"],
            "business_address": ["12 Main St", None, pd.NA, float("nan")],
            "country": ["US", "India", "US", "France"],
        }
    )
    secondary_columns_to_check = [
        "business_name_normalized", "business_name_alnum", "business_name_tokens", "business_name_token_sorted",
        "business_address_normalized", "business_address_alnum", "business_address_tokens", "business_address_token_sorted",
        "country_normalized",
    ]
    try:
        missing_result_df = add_normalized_columns(missing_address_df)
        for col in secondary_columns_to_check:
            for value in missing_result_df[col].tolist():
                is_missing_like = value is None or (isinstance(value, float) and pd.isna(value))
                is_ok = is_missing_like or isinstance(value, str)
                check(is_ok, f"Column '{col}' produced an unexpected non-string, non-missing value: {value!r} (type {type(value)})")
                if isinstance(value, str):
                    check(
                        value.strip().lower() not in ("nan", "none", "null", "missing", "<na>", "<nat>"),
                        f"Column '{col}' must not contain a literal missing-like string, got {value!r}",
                    )
        # Rows 1, 2, 3 (index 1,2,3) have missing business_address (None,
        # pd.NA, float NaN respectively) -> every derived address column
        # for those rows must be missing, not a crash and not "nan".
        for row_idx in (1, 2, 3):
            for col in ("business_address_normalized", "business_address_alnum", "business_address_tokens", "business_address_token_sorted"):
                addr_val = missing_result_df.loc[row_idx, col]
                is_missing_like = addr_val is None or (isinstance(addr_val, float) and pd.isna(addr_val))
                check(is_missing_like, f"Row {row_idx} with missing business_address must yield missing '{col}', got {addr_val!r}")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"add_normalized_columns() must handle missing business_address values (None/pd.NA/float NaN) without raising: {exc}")

    return failures


# ---------------------------------------------------------------------------
# Demonstration (small, in-memory, representative strings only)
# ---------------------------------------------------------------------------

def run_demo() -> pd.DataFrame:
    """
    Build and display a small in-memory demonstration DataFrame using
    representative strings drawn from patterns observed in the audit
    report, including a row with a missing business_address. Does NOT
    load any full dataset file. This calls the exact same
    add_normalized_columns() function that validate_normalization()
    already exercised with missing values above, so a passing validation
    step is a real guarantee about this demo's behavior.
    """
    demo_raw = pd.DataFrame(
        {
            "entity_id": ["S1-988243351", "S2-63862338", "S1-567455741", "S2-762236657", "S1-905686151"],
            "business_name": [
                "Orthopedic Associates Inc",
                "Orthopedic Inc Associates",
                "White Producer Private Limited",
                "व्हाइट प्रोड्यूसर प्राइवेट लिमिटेड",
                "Gertie's Colonial Holdings",
            ],
            "business_address": [
                "5101 Disbrow Court, Lincoln, NE",
                "DISBROW COURT, LINCON, NE",
                "House No 418, Gandhi Ganj, Hapur, Ghaziabad, Uttar Pradesh",
                "NO 163 HOUSE NO 418, GHAZIABAD, GANDHI GANJ, Uttar Pradesh",
                None,
            ],
            "country": ["US", "US", "India", "India", "US"],
        }
    )

    result = add_normalized_columns(demo_raw)
    display_cols = [
        "business_name",
        "business_name_normalized",
        "business_name_alnum",
        "business_name_token_sorted",
        "business_address",
        "business_address_normalized",
        "business_address_alnum",
        "country",
        "country_normalized",
    ]
    return result[display_cols]


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: conservative, deterministic preprocessing for Business Entity Resolution."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run validation checks and print a small in-memory demonstration, then exit.",
    )
    parser.add_argument(
        "--file",
        choices=sorted(SOURCE_FILES.keys()),
        help="Process exactly one named source file end-to-end using chunked streaming.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
        help=f"Rows per chunk when streaming a file (default: {DEFAULT_CHUNKSIZE}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write preprocessed TSVs into (default: {DEFAULT_OUTPUT_DIR}).",
    )
    args = parser.parse_args()

    if args.demo:
        failures = validate_normalization()
        if failures:
            print("VALIDATION FAILED. Issues found:")
            for msg in failures:
                print(f"  - {msg}")
            sys.exit(1)
        print("Validation checks: ALL PASSED.")
        print()
        demo_result = run_demo()
        with pd.option_context("display.max_columns", None, "display.width", 200):
            print(demo_result.to_string(index=False))
        return

    if args.file:
        input_path = SOURCE_FILES[args.file]
        output_path = os.path.join(args.output_dir, f"{args.file}_preprocessed.tsv")
        print(f"Processing '{args.file}' from: {input_path}")
        print(f"Writing preprocessed output to: {output_path}")
        print(f"Chunk size: {args.chunksize} rows")
        try:
            total_rows = process_file_in_chunks(input_path, output_path, chunksize=args.chunksize)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        except ValueError as exc:
            print(f"ERROR: Schema problem while processing '{args.file}': {exc}", file=sys.stderr)
            sys.exit(1)
        print(f"Done. Total rows processed: {total_rows}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()