"""
dataset_deep_profile.py

Amazon ML Challenge 2026 - Business Entity Resolution
COMPLETE-DATASET, memory-safe, analysis-only deep profiling script.

SCOPE (STRICT)
--------------
This script performs ONLY read-only profiling and bounded sampling of the
provided TSV datasets. It does NOT modify any input file, normalize or
transliterate any value, perform fuzzy matching, blocking, candidate
generation, feature engineering for a model, ML training, prediction, or
any form of entity resolution decision-making. It does not use external
data, APIs, or the internet.

This script reads ONLY the local files that already exist at the paths
below, resolved relative to the actual project root at runtime. It never
creates, replaces, or fabricates any dataset file, and it performs no
analysis unless those real files are present on disk when you run it.

MEMORY SAFETY
-------------
Source files and the ground-truth file can contain millions of rows.
This script NEVER loads a full source file or the full ground-truth file
into memory, NEVER concatenates all chunks (`pd.concat(all_chunks)`),
NEVER stores every row, and NEVER stores every unique
business_name/business_address/source1_entity_id value (cardinality
could be in the millions). All statistics are computed with streaming
counters, bounded reservoirs (deterministic reservoir sampling, seed=42),
and bounded dictionaries capped at 100,000 distinct tracked keys. Where an
exact global statistic cannot be computed safely without unbounded
memory, this is stated explicitly in the report as a LOWER BOUND / PARTIAL
result rather than silently presented as exact.

Run from the Amazon_ML_2026 project root:
    python Code/business_entity_resolution/src/dataset_deep_profile.py
    python Code/business_entity_resolution/src/dataset_deep_profile.py --chunksize 200000
    python Code/business_entity_resolution/src/dataset_deep_profile.py --output-dir output
"""

from __future__ import annotations

import os
import sys
import gc
import json
import random
import argparse
import unicodedata
from collections import Counter, defaultdict
from typing import Optional, List, Dict, Tuple, Set

import pandas as pd

# ---------------------------------------------------------------------------
# Paths (resolved at runtime relative to THIS file's location; no dataset
# file is created, replaced, or assumed to exist ahead of time - the
# script simply looks here for your real local files when you run it).
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASET_TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "train")
DATASET_TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "test")

SOURCE_FILES = {
    "train_source1": os.path.join(DATASET_TRAIN_DIR, "train_source1.tsv"),
    "train_source2": os.path.join(DATASET_TRAIN_DIR, "train_source2.tsv"),
    "train_source3": os.path.join(DATASET_TRAIN_DIR, "train_source3.tsv"),
    "test_source1": os.path.join(DATASET_TEST_DIR, "test_source1.tsv"),
    "test_source2": os.path.join(DATASET_TEST_DIR, "test_source2.tsv"),
    "test_source3": os.path.join(DATASET_TEST_DIR, "test_source3.tsv"),
}
GROUND_TRUTH_FILE = os.path.join(DATASET_TRAIN_DIR, "train_ground_truth.tsv")

REQUIRED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
REQUIRED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

DEFAULT_CHUNKSIZE = 100_000
GT_CHUNKSIZE = 100_000
RANDOM_SEED = 42
LENGTH_RESERVOIR_CAPACITY = 50_000
MULTILINGUAL_SAMPLE_PER_SCRIPT_FIELD = 20
TRUE_MATCH_SAMPLE_TARGET_PER_SOURCE = 10_000
TRUE_MATCH_EXAMPLES_OUTPUT_CAP = 500

# CRITICAL FIX 1: bounded duplicate tracking cap reduced from 2,000,000 to
# 100,000 distinct tracked keys per tracker. With six SourceFileProfile
# objects each holding three trackers (entity_id, business_name,
# business_address) plus the ground-truth source1_entity_id tracker, the
# previous cap could retain tens of millions of Python strings across all
# trackers simultaneously. 100,000 keeps total worst-case memory bounded
# and small, at the cost of duplicate counts becoming a LOWER BOUND /
# PARTIAL result (clearly labeled as such in every report section) once
# a tracker's cardinality exceeds the cap.
DUPLICATE_VALUE_TRACK_CAP = 100_000

UNICODE_CHAR_DIAGNOSTIC_TOP_N = 100
UNICODE_CHAR_EXAMPLES_PER_CATEGORY = 15

# ---------------------------------------------------------------------------
# Unicode script classification (stdlib only; explicit codepoint ranges)
#
# CRITICAL FIX 6: the previous single "CJK" bucket merged Han ideographs,
# Hiragana/Katakana, and Hangul into one category, which incorrectly
# treated three linguistically and script-wise distinct writing systems
# as equivalent. These are now three separate buckets: "Han" (CJK
# ideographs), "Hiragana_Katakana" (Japanese syllabaries), and "Hangul"
# (Korean). All existing Indian-script ranges (Devanagari, Gurmukhi,
# Bengali, Gujarati, Tamil, Telugu, Kannada, Malayalam, Odia) and all
# other existing categories are unchanged.
# ---------------------------------------------------------------------------
SCRIPT_RANGES: Dict[str, List[Tuple[int, int]]] = {
    "Latin": [(0x0041, 0x024F), (0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF)],
    "Devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "Gurmukhi": [(0x0A00, 0x0A7F)],
    "Bengali": [(0x0980, 0x09FF)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Odia": [(0x0B00, 0x0B7F)],
    "Arabic": [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "Hebrew": [(0x0590, 0x05FF)],
    "Cyrillic": [(0x0400, 0x04FF), (0x0500, 0x052F)],
    "Greek": [(0x0370, 0x03FF), (0x1F00, 0x1FFF)],
    "Han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)],
    "Hiragana_Katakana": [(0x3040, 0x30FF)],
    "Hangul": [(0xAC00, 0xD7A3), (0x1100, 0x11FF)],
    "Thai": [(0x0E00, 0x0E7F)],
    "Sinhala": [(0x0D80, 0x0DFF)],
}
SCRIPT_NAMES_ORDERED = list(SCRIPT_RANGES.keys()) + ["Other"]


def classify_char_script(ch: str) -> Optional[str]:
    """
    Classify a single character into a script bucket using explicit
    Unicode codepoint ranges (stdlib-only, no external Unicode database
    beyond what `unicodedata` already provides). Returns None for
    characters that carry no script signal (whitespace, punctuation,
    symbols, digits, and standalone combining marks with no matched
    range) since those should never register as a "script". Returns
    "Other" ONLY for an actual Letter-category character that does not
    fall into any explicitly known script range, so unanticipated
    writing systems are still counted and flagged - but digits (Nd) and
    combining marks (Mn/Mc) are explicitly excluded from the "Other"
    fallback, because a digit like "5" appearing in almost every address
    is not evidence of an unclassified script and must not inflate that
    bucket.
    """
    cp = ord(ch)
    for script, ranges in SCRIPT_RANGES.items():
        for lo, hi in ranges:
            if lo <= cp <= hi:
                return script
    category = unicodedata.category(ch)
    if category[0] == "L":
        return "Other"
    return None


def detect_scripts_in_text(text: str) -> Set[str]:
    """Return the set of distinct script buckets present in `text`."""
    found: Set[str] = set()
    for ch in text:
        script = classify_char_script(ch)
        if script:
            found.add(script)
    return found


# ---------------------------------------------------------------------------
# Missing-value helper (read-only profiling; never mutates or coerces)
# ---------------------------------------------------------------------------

def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    return False


# ---------------------------------------------------------------------------
# Deterministic reservoir sampling (Algorithm R)
# ---------------------------------------------------------------------------

class ReservoirSampler:
    """
    Bounded-memory, deterministic reservoir sampler. Given a fixed seed,
    the same stream of .add() calls (in the same order) always yields the
    same sample - required for reproducible profiling runs.
    """

    def __init__(self, capacity: int, seed: int = RANDOM_SEED):
        self.capacity = capacity
        self._rng = random.Random(seed)
        self.items: list = []
        self.n_seen = 0

    def add(self, item) -> None:
        self.n_seen += 1
        if len(self.items) < self.capacity:
            self.items.append(item)
        else:
            j = self._rng.randint(0, self.n_seen - 1)
            if j < self.capacity:
                self.items[j] = item

    def is_exact(self) -> bool:
        """True if every item ever added is still in the sample (no eviction)."""
        return self.n_seen <= self.capacity


# ---------------------------------------------------------------------------
# Streaming length statistics (min/max/mean exact; percentiles via reservoir)
# ---------------------------------------------------------------------------

class StreamingLengthStats:
    """
    Tracks exact count/min/max/mean in O(1) memory, and approximates
    percentiles via a bounded reservoir sample of observed lengths.
    Percentiles are EXACT if total observations <= reservoir capacity,
    and are clearly labeled as APPROXIMATE (reservoir-based) otherwise.
    """

    def __init__(self, reservoir_capacity: int = LENGTH_RESERVOIR_CAPACITY, seed: int = RANDOM_SEED):
        self.count = 0
        self.min_val: Optional[int] = None
        self.max_val: Optional[int] = None
        self.sum_val = 0
        self._reservoir = ReservoirSampler(reservoir_capacity, seed=seed)

    def add(self, length: int) -> None:
        self.count += 1
        self.sum_val += length
        if self.min_val is None or length < self.min_val:
            self.min_val = length
        if self.max_val is None or length > self.max_val:
            self.max_val = length
        self._reservoir.add(length)

    def summary(self) -> Optional[dict]:
        if self.count == 0:
            return None
        mean = self.sum_val / self.count
        sample_sorted = sorted(self._reservoir.items)

        def pct(p: float) -> Optional[float]:
            if not sample_sorted:
                return None
            idx = min(int(p / 100.0 * len(sample_sorted)), len(sample_sorted) - 1)
            return sample_sorted[idx]

        return {
            "count": self.count,
            "min": self.min_val,
            "max": self.max_val,
            "mean": round(mean, 2),
            "percentiles_exact": self._reservoir.is_exact(),
            "p25": pct(25), "p50_median": pct(50), "p75": pct(75),
            "p90": pct(90), "p95": pct(95), "p99": pct(99),
        }


# ---------------------------------------------------------------------------
# Bounded duplicate-value tracker
# ---------------------------------------------------------------------------

class BoundedDuplicateTracker:
    """
    Tracks how many values have been seen more than once, using a bounded
    dictionary of value -> count. If the number of distinct keys exceeds
    `max_distinct_keys` (default DUPLICATE_VALUE_TRACK_CAP = 100,000), new
    keys stop being added (existing tracked keys are still updated) and
    the tracker marks itself `capped`, so the final duplicate count is
    reported as a LOWER BOUND / PARTIAL result rather than silently
    presented as exact. This keeps memory bounded even if cardinality
    turns out to be in the millions.
    """

    def __init__(self, max_distinct_keys: int = DUPLICATE_VALUE_TRACK_CAP):
        self.max_distinct_keys = max_distinct_keys
        self._counts: Dict[str, int] = {}
        self.capped = False
        self.total_seen = 0
        self.total_skipped_due_to_cap = 0

    def add(self, value: str) -> None:
        self.add_batch(value, 1)

    def add_batch(self, value: str, count: int) -> None:
        """Record `count` additional occurrences of `value` in one call."""
        self.total_seen += count
        if value in self._counts:
            self._counts[value] += count
        else:
            if len(self._counts) < self.max_distinct_keys:
                self._counts[value] = count
            else:
                self.capped = True
                self.total_skipped_due_to_cap += count

    def duplicate_count(self) -> int:
        """Number of rows beyond the first occurrence, among TRACKED values."""
        return sum(c - 1 for c in self._counts.values() if c > 1)

    def distinct_tracked(self) -> int:
        return len(self._counts)

    def status_label(self) -> str:
        """Human-readable label to append to any reported statistic using this tracker."""
        return " [LOWER BOUND / PARTIAL - tracking capped at {:,} distinct keys]".format(self.max_distinct_keys) if self.capped else ""


# ---------------------------------------------------------------------------
# Text/noise characteristic checks (raw, non-normalizing)
# ---------------------------------------------------------------------------

_UNUSUAL_WHITESPACE_CODEPOINTS = {
    0x00A0,  # NO-BREAK SPACE
    0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
    0x2007, 0x2008, 0x2009, 0x200A, 0x202F, 0x205F, 0x3000,
}
_ZERO_WIDTH_CODEPOINTS = {0x200B, 0x200C, 0x200D, 0xFEFF}


def has_digit(text: str) -> bool:
    return any(unicodedata.category(ch) == "Nd" for ch in text)


def has_punctuation(text: str) -> bool:
    return any(unicodedata.category(ch).startswith("P") for ch in text)


def has_symbol(text: str) -> bool:
    return any(unicodedata.category(ch).startswith("S") for ch in text)


def has_non_ascii(text: str) -> bool:
    return any(ord(ch) > 127 for ch in text)


def has_leading_whitespace(text: str) -> bool:
    return len(text) > 0 and text[0].isspace()


def has_trailing_whitespace(text: str) -> bool:
    return len(text) > 0 and text[-1].isspace()


def has_repeated_whitespace(text: str) -> bool:
    prev_space = False
    for ch in text:
        if ch.isspace():
            if prev_space:
                return True
            prev_space = True
        else:
            prev_space = False
    return False


def has_mixed_case_latin(text: str) -> bool:
    has_upper = any(ch.isupper() and classify_char_script(ch) == "Latin" for ch in text)
    has_lower = any(ch.islower() and classify_char_script(ch) == "Latin" for ch in text)
    return has_upper and has_lower


def has_accented_latin(text: str) -> bool:
    for ch in text:
        if classify_char_script(ch) == "Latin":
            decomposed = unicodedata.decomposition(ch)
            if decomposed:
                return True
    return False


def has_unusual_whitespace(text: str) -> bool:
    return any(ord(ch) in _UNUSUAL_WHITESPACE_CODEPOINTS for ch in text)


def has_zero_width_char(text: str) -> bool:
    return any(ord(ch) in _ZERO_WIDTH_CODEPOINTS for ch in text)


def has_combining_mark(text: str) -> bool:
    return any(unicodedata.category(ch) in ("Mn", "Mc") for ch in text)


def has_fullwidth_char(text: str) -> bool:
    for ch in text:
        try:
            name = unicodedata.name(ch, "")
        except ValueError:
            name = ""
        if "FULLWIDTH" in name:
            return True
    return False


def has_compatibility_char(text: str) -> bool:
    for ch in text:
        decomp = unicodedata.decomposition(ch)
        if decomp and decomp.startswith("<"):
            return True
    return False


# ---------------------------------------------------------------------------
# Field-level streaming profile accumulator (one per file+field)
# ---------------------------------------------------------------------------

class FieldProfile:
    """
    Accumulates ALL streaming statistics for a single text field
    (business_name or business_address) within a single source file:
    missingness, length stats, script presence, mixed-script rate, and
    the various noise-pattern flags. Everything here is O(1)-per-row and
    bounded in total memory (reservoirs + fixed, capped counters only).
    """

    def __init__(self, field_name: str):
        self.field_name = field_name
        self.non_missing_count = 0
        self.missing_count = 0
        self.length_stats = StreamingLengthStats()
        self.script_row_counts: Counter = Counter()
        self.mixed_script_row_count = 0
        self.noise_counts: Counter = Counter()
        self.duplicate_tracker = BoundedDuplicateTracker()

    def to_dict(self) -> dict:
        total = self.non_missing_count + self.missing_count
        return {
            "field": self.field_name,
            "missing_count": self.missing_count,
            "missing_pct": round(100.0 * self.missing_count / total, 4) if total else None,
            "non_missing_count": self.non_missing_count,
            "length_stats": self.length_stats.summary(),
            "script_presence": {
                script: {
                    "rows_containing": count,
                    "pct_of_non_missing": round(100.0 * count / self.non_missing_count, 4) if self.non_missing_count else None,
                }
                for script, count in sorted(self.script_row_counts.items(), key=lambda kv: -kv[1])
            },
            "mixed_script_rows": self.mixed_script_row_count,
            "mixed_script_pct_of_non_missing": round(100.0 * self.mixed_script_row_count / self.non_missing_count, 4) if self.non_missing_count else None,
            "noise_patterns": {
                k: {
                    "count": v,
                    "pct_of_non_missing": round(100.0 * v / self.non_missing_count, 4) if self.non_missing_count else None,
                }
                for k, v in sorted(self.noise_counts.items())
            },
            "duplicate_raw_values": {
                "duplicate_count_among_tracked": self.duplicate_tracker.duplicate_count(),
                "distinct_tracked": self.duplicate_tracker.distinct_tracked(),
                "tracking_capped": self.duplicate_tracker.capped,
            },
        }


# ---------------------------------------------------------------------------
# Unicode character diagnostics (bounded top-N, plus representative examples)
# ---------------------------------------------------------------------------

class UnicodeCharDiagnostics:
    """
    Tracks a BOUNDED count of distinct codepoints (capped dictionary, same
    strategy as BoundedDuplicateTracker) across all profiled text, plus a
    small reservoir of representative example strings for specific
    categories of interest (combining marks, zero-width chars, unusual
    whitespace, full-width chars, non-ASCII punctuation, Indic vowel
    signs/marks). Never stores every character occurrence individually
    beyond the bounded counters.
    """

    def __init__(self, max_distinct_codepoints: int = 20_000):
        self.codepoint_counts: Dict[int, int] = {}
        self.capped = False
        self.max_distinct_codepoints = max_distinct_codepoints
        self.examples: Dict[str, ReservoirSampler] = {
            "combining_marks": ReservoirSampler(UNICODE_CHAR_EXAMPLES_PER_CATEGORY),
            "zero_width_chars": ReservoirSampler(UNICODE_CHAR_EXAMPLES_PER_CATEGORY),
            "unusual_whitespace": ReservoirSampler(UNICODE_CHAR_EXAMPLES_PER_CATEGORY),
            "fullwidth_chars": ReservoirSampler(UNICODE_CHAR_EXAMPLES_PER_CATEGORY),
            "non_ascii_punctuation": ReservoirSampler(UNICODE_CHAR_EXAMPLES_PER_CATEGORY),
            "indic_vowel_signs": ReservoirSampler(UNICODE_CHAR_EXAMPLES_PER_CATEGORY),
        }

    def process_value(self, value: str, source_file: str, field: str, entity_id: str) -> None:
        for ch in value:
            cp = ord(ch)
            if cp in self.codepoint_counts:
                self.codepoint_counts[cp] += 1
            elif len(self.codepoint_counts) < self.max_distinct_codepoints:
                self.codepoint_counts[cp] = 1
            else:
                self.capped = True

            category = unicodedata.category(ch)
            example_record = {"source_file": source_file, "field": field, "entity_id": entity_id, "value": value, "char": ch, "codepoint": f"U+{cp:04X}"}

            if category in ("Mn", "Mc"):
                self.examples["combining_marks"].add(example_record)
                script = classify_char_script(ch)
                if script in ("Devanagari", "Gurmukhi", "Bengali", "Gujarati", "Tamil", "Telugu", "Kannada", "Malayalam", "Odia"):
                    self.examples["indic_vowel_signs"].add(example_record)
            if cp in _ZERO_WIDTH_CODEPOINTS:
                self.examples["zero_width_chars"].add(example_record)
            if cp in _UNUSUAL_WHITESPACE_CODEPOINTS:
                self.examples["unusual_whitespace"].add(example_record)
            try:
                uname = unicodedata.name(ch, "")
            except ValueError:
                uname = ""
            if "FULLWIDTH" in uname:
                self.examples["fullwidth_chars"].add(example_record)
            if category.startswith("P") and cp > 127:
                self.examples["non_ascii_punctuation"].add(example_record)

    def top_n_summary(self, n: int = UNICODE_CHAR_DIAGNOSTIC_TOP_N) -> list:
        top = sorted(self.codepoint_counts.items(), key=lambda kv: -kv[1])[:n]
        result = []
        for cp, count in top:
            ch = chr(cp)
            try:
                name = unicodedata.name(ch, "UNKNOWN")
            except ValueError:
                name = "UNKNOWN"
            result.append({
                "codepoint": f"U+{cp:04X}",
                "character": ch if unicodedata.category(ch)[0] not in ("C",) else "<control>",
                "unicode_name": name,
                "category": unicodedata.category(ch),
                "count": count,
            })
        return result

    def examples_summary(self) -> dict:
        return {key: sampler.items for key, sampler in self.examples.items()}


# ---------------------------------------------------------------------------
# Per-source-file profiling (streaming, chunked, memory-safe)
# ---------------------------------------------------------------------------

class SourceFileProfile:
    """Holds all accumulated profiling state for one source TSV file."""

    def __init__(self, key: str, path: str):
        self.key = key
        self.path = path
        self.total_rows = 0
        self.chunks_processed = 0
        self.entity_id_tracker = BoundedDuplicateTracker()
        self.country_counts: Counter = Counter()
        self.country_missing_count = 0
        self.name_profile = FieldProfile("business_name")
        self.address_profile = FieldProfile("business_address")
        self.schema_ok = False
        self.error: Optional[str] = None
        self.file_complete = False
        self.multilingual_samplers: Dict[Tuple[str, str], ReservoirSampler] = defaultdict(
            lambda: ReservoirSampler(MULTILINGUAL_SAMPLE_PER_SCRIPT_FIELD)
        )
        self.unicode_diag = UnicodeCharDiagnostics()

    def to_dict(self) -> dict:
        total = self.total_rows
        return {
            "file_key": self.key,
            "path": self.path,
            "file_complete": self.file_complete,
            "schema_ok": self.schema_ok,
            "error": self.error,
            "total_rows": self.total_rows,
            "chunks_processed": self.chunks_processed,
            "unique_entity_id_tracked": self.entity_id_tracker.distinct_tracked(),
            "duplicate_entity_id_count": self.entity_id_tracker.duplicate_count(),
            "entity_id_tracking_capped": self.entity_id_tracker.capped,
            "country_missing_count": self.country_missing_count,
            "country_missing_pct": round(100.0 * self.country_missing_count / total, 4) if total else None,
            "country_counts": [
                {"country": c, "count": n, "pct": round(100.0 * n / total, 4) if total else None}
                for c, n in sorted(self.country_counts.items(), key=lambda kv: -kv[1])
            ],
            "business_name_profile": self.name_profile.to_dict(),
            "business_address_profile": self.address_profile.to_dict(),
        }


def verify_source_schema(columns: list, key: str) -> Optional[str]:
    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in columns]
    if missing:
        return f"Missing required columns {missing} in {key}; found columns: {columns}"
    return None


def _accumulate_noise_flags(noise_counter: Counter, value: str) -> None:
    if has_digit(value):
        noise_counter["contains_digit"] += 1
    if has_punctuation(value):
        noise_counter["contains_punctuation"] += 1
    if has_symbol(value):
        noise_counter["contains_symbol"] += 1
    if has_non_ascii(value):
        noise_counter["contains_non_ascii"] += 1
    if has_leading_whitespace(value):
        noise_counter["leading_whitespace"] += 1
    if has_trailing_whitespace(value):
        noise_counter["trailing_whitespace"] += 1
    if has_repeated_whitespace(value):
        noise_counter["repeated_whitespace"] += 1
    if has_mixed_case_latin(value):
        noise_counter["mixed_case_latin"] += 1
    if has_accented_latin(value):
        noise_counter["accented_latin"] += 1
    if has_unusual_whitespace(value):
        noise_counter["unusual_whitespace"] += 1
    if has_zero_width_char(value):
        noise_counter["zero_width_char"] += 1
    if has_combining_mark(value):
        noise_counter["combining_mark"] += 1
    if has_fullwidth_char(value):
        noise_counter["fullwidth_char"] += 1
    if has_compatibility_char(value):
        noise_counter["compatibility_char"] += 1


def profile_source_file(key: str, path: str, chunksize: int) -> SourceFileProfile:
    """
    Streams `path` in chunks of `chunksize` rows. Vectorized pandas
    operations (value_counts, isna, str.len) are used wherever possible
    for speed; genuinely per-character operations (script detection,
    noise-pattern flags, Unicode diagnostics, multilingual sampling) use
    a single itertuples() pass per chunk - itertuples() is far faster
    than iterrows() and does not materialize row Series objects, keeping
    both speed and memory bounded. No chunk is ever retained after
    processing, and no cross-chunk concatenation occurs. This function
    only reads `path`; it never writes to or otherwise modifies it.
    """
    profile = SourceFileProfile(key, path)

    if not os.path.exists(path):
        profile.error = f"File not found: {path}"
        return profile

    try:
        reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True, chunksize=chunksize)
        first_chunk = True
        for chunk in reader:
            if first_chunk:
                schema_error = verify_source_schema(list(chunk.columns), key)
                if schema_error:
                    profile.error = schema_error
                    return profile
                profile.schema_ok = True
                first_chunk = False

            entity_id_counts_this_chunk = chunk["entity_id"].dropna().value_counts()
            for eid, cnt in entity_id_counts_this_chunk.items():
                profile.entity_id_tracker.add_batch(str(eid), int(cnt))

            country_na_mask = chunk["country"].isna()
            profile.country_missing_count += int(country_na_mask.sum())
            country_value_counts = chunk.loc[~country_na_mask, "country"].value_counts()
            for country_val, cnt in country_value_counts.items():
                profile.country_counts[str(country_val)] += int(cnt)

            name_na_mask = chunk["business_name"].isna()
            addr_na_mask = chunk["business_address"].isna()
            profile.name_profile.missing_count += int(name_na_mask.sum())
            profile.address_profile.missing_count += int(addr_na_mask.sum())

            name_non_missing = chunk.loc[~name_na_mask, "business_name"]
            addr_non_missing = chunk.loc[~addr_na_mask, "business_address"]
            profile.name_profile.non_missing_count += len(name_non_missing)
            profile.address_profile.non_missing_count += len(addr_non_missing)

            for length in name_non_missing.str.len().tolist():
                profile.name_profile.length_stats.add(int(length))
            for length in addr_non_missing.str.len().tolist():
                profile.address_profile.length_stats.add(int(length))

            for row in chunk.itertuples(index=False):
                entity_id = getattr(row, "entity_id", None)
                name_val = getattr(row, "business_name", None)
                addr_val = getattr(row, "business_address", None)

                if not is_missing(name_val):
                    name_str = str(name_val)
                    profile.name_profile.duplicate_tracker.add(name_str)
                    scripts = detect_scripts_in_text(name_str)
                    for script in scripts:
                        profile.name_profile.script_row_counts[script] += 1
                        if script != "Latin":
                            profile.multilingual_samplers[("business_name", script)].add(
                                {"source_file": key, "field": "business_name", "script": script,
                                 "entity_id": entity_id, "value": name_str}
                            )
                    if len(scripts) > 1:
                        profile.name_profile.mixed_script_row_count += 1
                    _accumulate_noise_flags(profile.name_profile.noise_counts, name_str)
                    profile.unicode_diag.process_value(name_str, key, "business_name", entity_id)

                if not is_missing(addr_val):
                    addr_str = str(addr_val)
                    profile.address_profile.duplicate_tracker.add(addr_str)
                    scripts = detect_scripts_in_text(addr_str)
                    for script in scripts:
                        profile.address_profile.script_row_counts[script] += 1
                        if script != "Latin":
                            profile.multilingual_samplers[("business_address", script)].add(
                                {"source_file": key, "field": "business_address", "script": script,
                                 "entity_id": entity_id, "value": addr_str}
                            )
                    if len(scripts) > 1:
                        profile.address_profile.mixed_script_row_count += 1
                    _accumulate_noise_flags(profile.address_profile.noise_counts, addr_str)
                    profile.unicode_diag.process_value(addr_str, key, "business_address", entity_id)

            profile.total_rows += len(chunk)
            profile.chunks_processed += 1
            del chunk, name_non_missing, addr_non_missing
            gc.collect()

        profile.file_complete = True

    except Exception as exc:  # noqa: BLE001
        profile.error = f"Exception while processing {path}: {exc}"
        profile.file_complete = False

    return profile


# ---------------------------------------------------------------------------
# Ground-truth streaming analysis
#
# The ground-truth file is processed in chunks of GT_CHUNKSIZE rows,
# exactly like the source files. No full ground-truth DataFrame is ever
# held in memory, and no Python list containing every matched ID across
# all rows is ever constructed. All match-count and prefix statistics are
# streaming counters; source1_entity_id uniqueness uses the same bounded
# BoundedDuplicateTracker (capped at DUPLICATE_VALUE_TRACK_CAP) as source
# files, so duplicate counts are reported as LOWER BOUND / PARTIAL if the
# cap is reached. True-match pair sampling uses two independent
# ReservoirSampler instances (seeded, capacity 10,000 each) updated
# incrementally while streaming - the reservoirs are the ONLY per-pair
# state retained, never a complete list of all ground-truth pairs.
# ---------------------------------------------------------------------------

def parse_matched_ids(raw_value) -> List[str]:
    """Split a matched_entity_ids cell into individual IDs. Analysis-only; does not mutate source."""
    if is_missing(raw_value):
        return []
    if not isinstance(raw_value, str):
        raw_value = str(raw_value)
    if raw_value == "":
        return []
    return [t for t in raw_value.split(",") if t != ""]


def bucket_match_count(n: int) -> str:
    return str(n) if n < 12 else "12+"


class GroundTruthProfile:
    def __init__(self):
        self.total_rows = 0
        self.chunks_processed = 0
        self.source1_id_tracker = BoundedDuplicateTracker()
        self.zero_match_count = 0
        self.one_match_count = 0
        self.multi_match_count = 0
        self.max_matches = 0
        self.match_count_distribution: Counter = Counter()
        self.total_matched_id_occurrences = 0
        self.s2_matched_occurrences = 0
        self.s3_matched_occurrences = 0
        self.invalid_prefix_occurrences = 0
        self.invalid_prefix_examples: list = []
        self.malformed_row_count = 0
        self.malformed_row_examples: list = []
        self.error: Optional[str] = None
        self.file_complete = False
        self.schema_ok = False
        self.s2_sampler = ReservoirSampler(TRUE_MATCH_SAMPLE_TARGET_PER_SOURCE, seed=RANDOM_SEED)
        self.s3_sampler = ReservoirSampler(TRUE_MATCH_SAMPLE_TARGET_PER_SOURCE, seed=RANDOM_SEED + 1)


def profile_ground_truth_streaming(path: str, chunksize: int) -> GroundTruthProfile:
    """
    Streams train_ground_truth.tsv in chunks of `chunksize` rows (default
    GT_CHUNKSIZE = 100,000), computing every required statistic with O(1)
    per-row streaming counters and two bounded reservoir samplers for
    true-match pairs. Never constructs a full ground-truth DataFrame or a
    list of all matched IDs. Read-only: never writes to `path`.
    """
    gt = GroundTruthProfile()

    if not os.path.exists(path):
        gt.error = f"Ground truth file not found: {path}"
        return gt

    try:
        reader = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True, chunksize=chunksize)
        first_chunk = True
        global_row_index = 0

        for chunk in reader:
            if first_chunk:
                missing_cols = [c for c in REQUIRED_GT_COLUMNS if c not in chunk.columns]
                if missing_cols:
                    gt.error = f"Ground truth missing required columns: {missing_cols}; found: {list(chunk.columns)}"
                    return gt
                gt.schema_ok = True
                first_chunk = False

            s1_series = chunk["source1_entity_id"]
            matched_series = chunk["matched_entity_ids"]

            for s1_id, raw_matched in zip(s1_series, matched_series):
                if is_missing(s1_id):
                    gt.malformed_row_count += 1
                    if len(gt.malformed_row_examples) < 20:
                        gt.malformed_row_examples.append({"row_index": global_row_index, "reason": "missing source1_entity_id"})
                    global_row_index += 1
                    continue

                s1_id_str = str(s1_id)
                gt.source1_id_tracker.add(s1_id_str)

                matched_list = parse_matched_ids(raw_matched)
                n_matches = len(matched_list)

                gt.match_count_distribution[n_matches] += 1
                if n_matches == 0:
                    gt.zero_match_count += 1
                elif n_matches == 1:
                    gt.one_match_count += 1
                else:
                    gt.multi_match_count += 1
                if n_matches > gt.max_matches:
                    gt.max_matches = n_matches

                gt.total_matched_id_occurrences += n_matches
                for mid in matched_list:
                    if mid.startswith("S2-"):
                        gt.s2_matched_occurrences += 1
                        gt.s2_sampler.add((s1_id_str, mid))
                    elif mid.startswith("S3-"):
                        gt.s3_matched_occurrences += 1
                        gt.s3_sampler.add((s1_id_str, mid))
                    else:
                        gt.invalid_prefix_occurrences += 1
                        if len(gt.invalid_prefix_examples) < 20:
                            gt.invalid_prefix_examples.append(mid)

                gt.total_rows += 1
                global_row_index += 1

            gt.chunks_processed += 1
            del chunk, s1_series, matched_series
            gc.collect()

        gt.file_complete = True

    except Exception as exc:  # noqa: BLE001
        gt.error = f"Exception while processing {path}: {exc}"
        gt.file_complete = False

    return gt


# ---------------------------------------------------------------------------
# True-match ID resolution (memory-safe, targeted streaming lookups)
#
# resolve_ids_from_source_streaming() never swallows exceptions silently.
# A read/parse failure while resolving sampled IDs is raised as a
# SourceResolutionError, which the caller (main()) handles explicitly - it
# is reported in the TXT report, the JSON summary, and reflected in the
# overall PROFILE COMPLETE / PROFILE FAILED exit status.
# ---------------------------------------------------------------------------

class SourceResolutionError(Exception):
    """Raised when streaming resolution of sampled true-match IDs from a source file fails."""
    pass


def resolve_ids_from_source_streaming(path: str, wanted_ids: Set[str], chunksize: int) -> Dict[str, dict]:
    """
    Stream `path` in chunks, retaining ONLY rows whose entity_id is in
    `wanted_ids`. Returns a bounded dict: entity_id -> row fields. Never
    materializes the full file; stops early once every wanted ID is found.
    Read-only: never writes to `path`.

    Raises SourceResolutionError if the file cannot be read/parsed at
    all. IDs that are simply not present in the file (a normal, expected
    outcome) are NOT an error - they are reported separately by the
    caller as "unresolved" IDs.
    """
    found: Dict[str, dict] = {}
    if not wanted_ids:
        return found
    if not os.path.exists(path):
        raise SourceResolutionError(f"Cannot resolve IDs: file not found: {path}")

    remaining = set(wanted_ids)
    try:
        for chunk in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=True, chunksize=chunksize):
            if "entity_id" not in chunk.columns:
                raise SourceResolutionError(f"Column 'entity_id' missing while resolving IDs from {path}")
            hit_mask = chunk["entity_id"].isin(remaining)
            if hit_mask.any():
                hits = chunk.loc[hit_mask]
                for row in hits.itertuples(index=False):
                    found[row.entity_id] = {
                        "business_name": getattr(row, "business_name", None),
                        "business_address": getattr(row, "business_address", None),
                        "country": getattr(row, "country", None),
                    }
                remaining -= set(hits["entity_id"].tolist())
            del chunk
            if not remaining:
                break
    except SourceResolutionError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise SourceResolutionError(f"Failed while streaming {path} to resolve sampled IDs: {exc}") from exc

    return found


# ---------------------------------------------------------------------------
# True-match comparison features (analysis-only; not used for matching)
# ---------------------------------------------------------------------------

def normalize_whitespace_only(text: str) -> str:
    return " ".join(text.split())


def strip_punctuation_only(text: str) -> str:
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def compare_names(name_a, name_b) -> dict:
    a_missing, b_missing = is_missing(name_a), is_missing(name_b)
    if a_missing or b_missing:
        return {"comparable": False, "either_missing": True}
    a, b = str(name_a), str(name_b)
    a_ws, b_ws = normalize_whitespace_only(a), normalize_whitespace_only(b)
    a_ci, b_ci = a_ws.casefold(), b_ws.casefold()
    a_punct, b_punct = strip_punctuation_only(a_ci), strip_punctuation_only(b_ci)
    a_tokens, b_tokens = set(a_punct.split()), set(b_punct.split())
    return {
        "comparable": True,
        "exact_raw_equal": a == b,
        "case_insensitive_equal": a_ci == b_ci,
        "whitespace_normalized_equal": a_ws == b_ws,
        "punctuation_insensitive_equal": a_punct == b_punct,
        "token_set_equal": a_tokens == b_tokens,
        "token_order_differs": (a_tokens == b_tokens) and (a_punct.split() != b_punct.split()),
        "length_diff": abs(len(a) - len(b)),
    }


def compare_addresses(addr_a, addr_b) -> dict:
    a_missing, b_missing = is_missing(addr_a), is_missing(addr_b)
    if a_missing or b_missing:
        return {"comparable": False, "either_missing": True, "missing_side": "both" if (a_missing and b_missing) else ("a" if a_missing else "b")}
    a, b = str(addr_a), str(addr_b)
    a_ws, b_ws = normalize_whitespace_only(a), normalize_whitespace_only(b)
    a_ci, b_ci = a_ws.casefold(), b_ws.casefold()
    a_punct, b_punct = strip_punctuation_only(a_ci), strip_punctuation_only(b_ci)
    return {
        "comparable": True,
        "either_missing": False,
        "exact_raw_equal": a == b,
        "case_insensitive_equal": a_ci == b_ci,
        "whitespace_normalized_equal": a_ws == b_ws,
        "punctuation_insensitive_equal": a_punct == b_punct,
        "length_diff": abs(len(a) - len(b)),
    }


def compare_countries(country_a, country_b) -> dict:
    a_missing, b_missing = is_missing(country_a), is_missing(country_b)
    if a_missing or b_missing:
        return {"comparable": False, "either_missing": True}
    return {"comparable": True, "exact_equal": str(country_a) == str(country_b), "either_missing": False}


# ---------------------------------------------------------------------------
# True-match example categorization (only created when data supports it)
#
# The "different_script" category compares the FULL detected script sets
# on each side (not just the non-Latin subset), so it correctly captures
# any script-set mismatch - including a Latin-only name on one side
# matched against a non-Latin name on the other (e.g. Latin vs.
# Devanagari), not only cases where BOTH sides are non-Latin and those
# non-Latin subsets differ. It does not inspect Unicode normalization
# forms (NFC/NFKC/NFD/NFKD) at all - it is purely a script-set comparison.
# ---------------------------------------------------------------------------

def categorize_match_example(s1_row: dict, matched_row: dict, name_cmp: dict, addr_cmp: dict, country_cmp: dict) -> List[str]:
    categories = []
    if name_cmp.get("comparable"):
        if name_cmp["exact_raw_equal"]:
            categories.append("exact_name_match")
        elif name_cmp["case_insensitive_equal"] and not name_cmp["exact_raw_equal"]:
            categories.append("case_difference")
        elif name_cmp["punctuation_insensitive_equal"] and not name_cmp["case_insensitive_equal"]:
            categories.append("punctuation_difference")
        elif name_cmp["whitespace_normalized_equal"] and not name_cmp["exact_raw_equal"]:
            categories.append("whitespace_difference")
        elif name_cmp.get("token_order_differs"):
            categories.append("token_reordering")
        elif not name_cmp["token_set_equal"]:
            categories.append("spelling_or_noise_difference")

    if addr_cmp.get("either_missing"):
        categories.append("missing_address")
    elif addr_cmp.get("comparable") and not addr_cmp["punctuation_insensitive_equal"]:
        if addr_cmp["length_diff"] > 15:
            categories.append("substantially_different_address")

    s1_name_scripts = detect_scripts_in_text(str(s1_row.get("business_name", "") or ""))
    m_name_scripts = detect_scripts_in_text(str(matched_row.get("business_name", "") or ""))
    s1_addr_scripts = detect_scripts_in_text(str(s1_row.get("business_address", "") or ""))
    m_addr_scripts = detect_scripts_in_text(str(matched_row.get("business_address", "") or ""))

    non_latin_s1_name = s1_name_scripts - {"Latin"}
    non_latin_m_name = m_name_scripts - {"Latin"}
    if non_latin_s1_name or non_latin_m_name:
        categories.append("multilingual_name")
    non_latin_s1_addr = s1_addr_scripts - {"Latin"}
    non_latin_m_addr = m_addr_scripts - {"Latin"}
    if non_latin_s1_addr or non_latin_m_addr:
        categories.append("multilingual_address")
    if len(s1_name_scripts) > 1 or len(m_name_scripts) > 1:
        categories.append("mixed_script_value")
    if s1_name_scripts != m_name_scripts and s1_name_scripts and m_name_scripts:
        categories.append("different_script")

    if country_cmp.get("comparable") and not country_cmp["exact_equal"]:
        categories.append("country_difference")

    return categories


# ---------------------------------------------------------------------------
# Report writing (text report)
# ---------------------------------------------------------------------------

class ReportWriter:
    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w", encoding="utf-8")

    def line(self, text: str = "") -> None:
        self._fh.write(str(text) + "\n")

    def header(self, text: str, level: int = 1) -> None:
        bar = "=" * 80 if level == 1 else "-" * 80
        self.line()
        self.line(bar)
        self.line(text)
        self.line(bar)

    def subheader(self, text: str) -> None:
        self.line()
        self.line(f"--- {text} ---")

    def close(self) -> None:
        self._fh.close()


def write_field_profile_section(rw: ReportWriter, file_key: str, field_dict: dict) -> None:
    rw.subheader(f"{file_key} :: {field_dict['field']}")
    rw.line(f"MEASURED STATISTIC - missing: {field_dict['missing_count']} ({field_dict['missing_pct']}%)")
    ls = field_dict["length_stats"]
    if ls:
        exact_label = "EXACT" if ls["percentiles_exact"] else "APPROXIMATE (reservoir-sampled)"
        rw.line(f"MEASURED STATISTIC - length: count={ls['count']} min={ls['min']} max={ls['max']} mean={ls['mean']}")
        rw.line(f"MEASURED STATISTIC - length percentiles [{exact_label}]: "
                f"p25={ls['p25']} p50={ls['p50_median']} p75={ls['p75']} p90={ls['p90']} p95={ls['p95']} p99={ls['p99']}")
    rw.line("MEASURED STATISTIC - script presence (row may contain multiple scripts, percentages need not sum to 100%):")
    for script, info in field_dict["script_presence"].items():
        rw.line(f"    {script}: {info['rows_containing']} rows ({info['pct_of_non_missing']}% of non-missing)")
    rw.line(f"MEASURED STATISTIC - mixed-script rows: {field_dict['mixed_script_rows']} ({field_dict['mixed_script_pct_of_non_missing']}% of non-missing)")
    rw.line("MEASURED STATISTIC - noise patterns:")
    for pattern, info in field_dict["noise_patterns"].items():
        rw.line(f"    {pattern}: {info['count']} ({info['pct_of_non_missing']}% of non-missing)")
    dup = field_dict["duplicate_raw_values"]
    capped_note = " [LOWER BOUND / PARTIAL - tracking capped due to memory limit]" if dup["tracking_capped"] else ""
    rw.line(f"MEASURED STATISTIC - duplicate raw values: {dup['duplicate_count_among_tracked']}{capped_note}")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Deep, memory-safe, complete-dataset profiling for Business Entity Resolution.")
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE, help=f"Rows per chunk for source files (default: {DEFAULT_CHUNKSIZE})")
    parser.add_argument("--output-dir", type=str, default=os.path.join(PROJECT_ROOT, "output"), help="Output directory (default: output)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "dataset_deep_profile.txt")
    json_path = os.path.join(output_dir, "dataset_deep_profile.json")
    multilingual_samples_path = os.path.join(output_dir, "representative_multilingual_samples.tsv")
    true_match_samples_path = os.path.join(output_dir, "representative_true_match_pairs.tsv")

    rw = ReportWriter(report_path)
    rw.header("AMAZON ML CHALLENGE 2026 - DATASET DEEP PROFILE (COMPLETE DATA)")
    rw.line("Analysis-only. No input file was modified. No matching/ML/prediction performed.")
    rw.line(f"Project root: {PROJECT_ROOT}")
    rw.line(f"Source file chunk size: {args.chunksize}")
    rw.line(f"Ground-truth chunk size: {GT_CHUNKSIZE}")
    rw.line(f"Duplicate-tracking cap: {DUPLICATE_VALUE_TRACK_CAP:,} distinct keys per tracker")

    json_summary: dict = {"files": {}, "ground_truth": {}, "true_match_sampling": {}, "processing_status": {}}

    rw.header("1. EXECUTIVE DATA SUMMARY")
    file_profiles: Dict[str, SourceFileProfile] = {}
    all_files_ok = True

    for key, path in SOURCE_FILES.items():
        print(f"Profiling {key} ...")
        profile = profile_source_file(key, path, args.chunksize)
        file_profiles[key] = profile
        status_ok = profile.file_complete and profile.error is None
        all_files_ok = all_files_ok and status_ok
        rw.line(f"{key}: rows={profile.total_rows}, chunks={profile.chunks_processed}, "
                f"complete={'YES' if profile.file_complete else 'NO'}, error={profile.error}")

    rw.header("2. TRAIN/TEST SOURCE STATISTICS", level=2)
    for key, profile in file_profiles.items():
        rw.subheader(f"File: {key} ({profile.path})")
        rw.line(f"MEASURED STATISTIC - total rows: {profile.total_rows}")
        rw.line(f"MEASURED STATISTIC - unique entity_id (tracked): {profile.entity_id_tracker.distinct_tracked()}"
                + profile.entity_id_tracker.status_label())
        rw.line(f"MEASURED STATISTIC - duplicate entity_id count: {profile.entity_id_tracker.duplicate_count()}"
                + profile.entity_id_tracker.status_label())

    rw.header("3. MISSING VALUES", level=2)
    for key, profile in file_profiles.items():
        rw.subheader(key)
        total = profile.total_rows
        rw.line(f"MEASURED STATISTIC - business_name missing: {profile.name_profile.missing_count} "
                f"({round(100.0*profile.name_profile.missing_count/total,4) if total else None}%)")
        rw.line(f"MEASURED STATISTIC - business_address missing: {profile.address_profile.missing_count} "
                f"({round(100.0*profile.address_profile.missing_count/total,4) if total else None}%)")
        rw.line(f"MEASURED STATISTIC - country missing: {profile.country_missing_count} "
                f"({round(100.0*profile.country_missing_count/total,4) if total else None}%)")

    rw.header("4. COUNTRY DISTRIBUTIONS", level=2)
    for key, profile in file_profiles.items():
        rw.subheader(key)
        total = profile.total_rows
        for country, count in sorted(profile.country_counts.items(), key=lambda kv: -kv[1]):
            pct = round(100.0 * count / total, 4) if total else None
            rw.line(f"MEASURED STATISTIC - '{country}': {count} ({pct}%)")

    rw.header("5. UNICODE/SCRIPT DISTRIBUTIONS", level=2)
    for key, profile in file_profiles.items():
        write_field_profile_section(rw, key, profile.name_profile.to_dict())
        write_field_profile_section(rw, key, profile.address_profile.to_dict())

    rw.header("6. MIXED-SCRIPT STATISTICS", level=2)
    for key, profile in file_profiles.items():
        rw.subheader(key)
        rw.line(f"MEASURED STATISTIC - business_name mixed-script rows: {profile.name_profile.mixed_script_row_count}")
        rw.line(f"MEASURED STATISTIC - business_address mixed-script rows: {profile.address_profile.mixed_script_row_count}")

    rw.header("7. UNICODE CHARACTER DIAGNOSTICS", level=2)
    for key, profile in file_profiles.items():
        rw.subheader(f"{key} - top {UNICODE_CHAR_DIAGNOSTIC_TOP_N} characters across business_name + business_address")
        for entry in profile.unicode_diag.top_n_summary():
            rw.line(f"MEASURED STATISTIC - {entry['codepoint']} '{entry['character']}' ({entry['unicode_name']}, {entry['category']}): {entry['count']}")
        rw.subheader(f"{key} - representative Unicode examples")
        for category, examples in profile.unicode_diag.examples_summary().items():
            rw.line(f"REPRESENTATIVE EXAMPLE category: {category} (showing up to {len(examples)})")
            for ex in examples[:10]:
                rw.line(f"    entity_id={ex['entity_id']} field={ex['field']} char={ex['char']!r} ({ex['codepoint']}) value={ex['value']!r}")

    rw.header("8. TEXT/NOISE STATISTICS", level=2)
    rw.line("(See section 5 above - noise pattern statistics are reported per file/field together with script stats.)")

    rw.header("9. DUPLICATE STATISTICS", level=2)
    for key, profile in file_profiles.items():
        rw.subheader(key)
        rw.line(f"MEASURED STATISTIC - duplicate entity_id: {profile.entity_id_tracker.duplicate_count()}"
                + profile.entity_id_tracker.status_label())
        rw.line(f"MEASURED STATISTIC - duplicate raw business_name: {profile.name_profile.duplicate_tracker.duplicate_count()}"
                + profile.name_profile.duplicate_tracker.status_label())
        rw.line(f"MEASURED STATISTIC - duplicate raw business_address: {profile.address_profile.duplicate_tracker.duplicate_count()}"
                + profile.address_profile.duplicate_tracker.status_label())

    print("Profiling ground truth (streaming, chunked) ...")
    gt = profile_ground_truth_streaming(GROUND_TRUTH_FILE, GT_CHUNKSIZE)
    gt_ok = gt.file_complete and gt.error is None
    all_files_ok = all_files_ok and gt_ok

    rw.header("10. GROUND-TRUTH MATCH DISTRIBUTION", level=2)
    if gt.error:
        rw.line(f"ERROR: {gt.error}")
    else:
        rw.line(f"MEASURED STATISTIC - total ground-truth rows: {gt.total_rows}")
        rw.line(f"MEASURED STATISTIC - chunks processed: {gt.chunks_processed}")
        rw.line(f"MEASURED STATISTIC - unique source1_entity_id (tracked): {gt.source1_id_tracker.distinct_tracked()}"
                + gt.source1_id_tracker.status_label())
        rw.line(f"MEASURED STATISTIC - duplicate source1_entity_id: {gt.source1_id_tracker.duplicate_count()}"
                + gt.source1_id_tracker.status_label())
        rw.line(f"MEASURED STATISTIC - zero-match: {gt.zero_match_count} ({round(100.0*gt.zero_match_count/gt.total_rows,4) if gt.total_rows else None}%)")
        rw.line(f"MEASURED STATISTIC - one-match: {gt.one_match_count} ({round(100.0*gt.one_match_count/gt.total_rows,4) if gt.total_rows else None}%)")
        rw.line(f"MEASURED STATISTIC - 2+ match: {gt.multi_match_count} ({round(100.0*gt.multi_match_count/gt.total_rows,4) if gt.total_rows else None}%)")
        rw.line(f"MEASURED STATISTIC - maximum matches for one Source-1 entity: {gt.max_matches}")
        rw.line(f"MEASURED STATISTIC - total matched-ID occurrences: {gt.total_matched_id_occurrences}")
        rw.line(f"MEASURED STATISTIC - total S2 matched-ID occurrences: {gt.s2_matched_occurrences}")
        rw.line(f"MEASURED STATISTIC - total S3 matched-ID occurrences: {gt.s3_matched_occurrences}")
        rw.line(f"MEASURED STATISTIC - invalid-prefix matched-ID occurrences: {gt.invalid_prefix_occurrences}")
        if gt.invalid_prefix_examples:
            rw.line(f"REPRESENTATIVE EXAMPLE - invalid-prefix IDs: {gt.invalid_prefix_examples}")
        if gt.malformed_row_count:
            rw.line(f"OBSERVATION - malformed ground-truth rows found: {gt.malformed_row_count}")
            for m in gt.malformed_row_examples:
                rw.line(f"    {m}")
        else:
            rw.line("OBSERVATION - no malformed ground-truth rows detected.")

        rw.subheader("Distribution of number_of_matches_per_source1_entity")
        bucket_totals: Dict[str, int] = defaultdict(int)
        for n_matches, n_rows in gt.match_count_distribution.items():
            bucket_totals[bucket_match_count(n_matches)] += n_rows
        for b in [str(i) for i in range(12)] + ["12+"]:
            count = bucket_totals.get(b, 0)
            pct = round(100.0 * count / gt.total_rows, 4) if gt.total_rows else None
            rw.line(f"MEASURED STATISTIC - {b} matches: {count} entities ({pct}%)")
        rw.line(f"OBSERVATION - dynamically detected true maximum matches: {gt.max_matches}")

    rw.header("11. TRUE-MATCH COMPARISON STATISTICS", level=2)
    true_match_rows_for_tsv: List[dict] = []
    multilingual_match_categories: Counter = Counter()
    resolution_error: Optional[str] = None

    if gt_ok:
        s2_sample = gt.s2_sampler.items
        s3_sample = gt.s3_sampler.items
        rw.line(f"OBSERVATION - sampled S1->S2 true-match pairs (bounded reservoir, seed={RANDOM_SEED}): {len(s2_sample)} (target {TRUE_MATCH_SAMPLE_TARGET_PER_SOURCE})")
        rw.line(f"OBSERVATION - sampled S1->S3 true-match pairs (bounded reservoir, seed={RANDOM_SEED + 1}): {len(s3_sample)} (target {TRUE_MATCH_SAMPLE_TARGET_PER_SOURCE})")

        s1_requested_ids = {s1 for s1, _ in s2_sample} | {s1 for s1, _ in s3_sample}
        s2_requested_ids = {mid for _, mid in s2_sample}
        s3_requested_ids = {mid for _, mid in s3_sample}

        s1_lookup: Dict[str, dict] = {}
        s2_lookup: Dict[str, dict] = {}
        s3_lookup: Dict[str, dict] = {}

        try:
            print(f"Resolving {len(s1_requested_ids)} sampled Source-1 records ...")
            s1_lookup = resolve_ids_from_source_streaming(SOURCE_FILES["train_source1"], s1_requested_ids, args.chunksize)
            print(f"Resolving {len(s2_requested_ids)} sampled Source-2 records ...")
            s2_lookup = resolve_ids_from_source_streaming(SOURCE_FILES["train_source2"], s2_requested_ids, args.chunksize)
            print(f"Resolving {len(s3_requested_ids)} sampled Source-3 records ...")
            s3_lookup = resolve_ids_from_source_streaming(SOURCE_FILES["train_source3"], s3_requested_ids, args.chunksize)
        except SourceResolutionError as exc:
            resolution_error = str(exc)
            rw.line(f"ERROR - true-match ID resolution failed and was NOT silently ignored: {resolution_error}")

        s1_resolved_ids = set(s1_lookup.keys())
        s2_resolved_ids = set(s2_lookup.keys())
        s3_resolved_ids = set(s3_lookup.keys())
        s1_unresolved_ids = s1_requested_ids - s1_resolved_ids
        s2_unresolved_ids = s2_requested_ids - s2_resolved_ids
        s3_unresolved_ids = s3_requested_ids - s3_resolved_ids

        rw.subheader("Sampled true-match ID resolution (requested / resolved / unresolved)")
        rw.line(f"MEASURED STATISTIC - Source-1 requested IDs: {len(s1_requested_ids)}")
        rw.line(f"MEASURED STATISTIC - Source-1 resolved IDs: {len(s1_resolved_ids)}")
        rw.line(f"MEASURED STATISTIC - Source-1 unresolved IDs: {len(s1_unresolved_ids)}")
        if s1_unresolved_ids:
            rw.line(f"REPRESENTATIVE EXAMPLE - unresolved Source-1 IDs (up to 20): {list(s1_unresolved_ids)[:20]}")
        rw.line(f"MEASURED STATISTIC - Source-2 requested IDs: {len(s2_requested_ids)}")
        rw.line(f"MEASURED STATISTIC - Source-2 resolved IDs: {len(s2_resolved_ids)}")
        rw.line(f"MEASURED STATISTIC - Source-2 unresolved IDs: {len(s2_unresolved_ids)}")
        if s2_unresolved_ids:
            rw.line(f"REPRESENTATIVE EXAMPLE - unresolved Source-2 IDs (up to 20): {list(s2_unresolved_ids)[:20]}")
        rw.line(f"MEASURED STATISTIC - Source-3 requested IDs: {len(s3_requested_ids)}")
        rw.line(f"MEASURED STATISTIC - Source-3 resolved IDs: {len(s3_resolved_ids)}")
        rw.line(f"MEASURED STATISTIC - Source-3 unresolved IDs: {len(s3_unresolved_ids)}")
        if s3_unresolved_ids:
            rw.line(f"REPRESENTATIVE EXAMPLE - unresolved Source-3 IDs (up to 20): {list(s3_unresolved_ids)[:20]}")

        name_cmp_agg: Counter = Counter()
        addr_cmp_agg: Counter = Counter()
        country_cmp_agg: Counter = Counter()
        n_compared = 0
        n_sampled_pairs_total = len(s2_sample) + len(s3_sample)
        example_rows_by_category: Dict[str, list] = defaultdict(list)

        for source_label, sample, lookup in (("S2", s2_sample, s2_lookup), ("S3", s3_sample, s3_lookup)):
            for s1_id, mid in sample:
                s1_row = s1_lookup.get(s1_id)
                m_row = lookup.get(mid)
                if s1_row is None or m_row is None:
                    continue
                n_compared += 1
                name_cmp = compare_names(s1_row["business_name"], m_row["business_name"])
                addr_cmp = compare_addresses(s1_row["business_address"], m_row["business_address"])
                country_cmp = compare_countries(s1_row["country"], m_row["country"])

                if name_cmp.get("comparable"):
                    for flag in ("exact_raw_equal", "case_insensitive_equal", "whitespace_normalized_equal",
                                 "punctuation_insensitive_equal", "token_set_equal", "token_order_differs"):
                        if name_cmp[flag]:
                            name_cmp_agg[f"name_{flag}"] += 1
                if addr_cmp.get("either_missing"):
                    addr_cmp_agg["address_either_missing"] += 1
                elif addr_cmp.get("comparable"):
                    for flag in ("exact_raw_equal", "case_insensitive_equal", "whitespace_normalized_equal", "punctuation_insensitive_equal"):
                        if addr_cmp[flag]:
                            addr_cmp_agg[f"address_{flag}"] += 1
                if country_cmp.get("comparable"):
                    if country_cmp["exact_equal"]:
                        country_cmp_agg["country_exact_equal"] += 1
                    else:
                        country_cmp_agg["country_differs"] += 1

                categories = categorize_match_example(s1_row, m_row, name_cmp, addr_cmp, country_cmp)
                for cat in categories:
                    if cat.startswith("multilingual") or cat in ("mixed_script_value", "different_script"):
                        multilingual_match_categories[cat] += 1
                    if len(example_rows_by_category[cat]) < 10:
                        example_rows_by_category[cat].append({
                            "source1_entity_id": s1_id, "matched_entity_id": mid, "matched_source": source_label,
                            "source1_business_name": s1_row["business_name"], "matched_business_name": m_row["business_name"],
                            "source1_business_address": s1_row["business_address"], "matched_business_address": m_row["business_address"],
                            "source1_country": s1_row["country"], "matched_country": m_row["country"],
                        })

                if len(true_match_rows_for_tsv) < TRUE_MATCH_EXAMPLES_OUTPUT_CAP:
                    true_match_rows_for_tsv.append({
                        "source1_entity_id": s1_id, "matched_entity_id": mid, "matched_source": source_label,
                        "source1_business_name": s1_row["business_name"], "matched_business_name": m_row["business_name"],
                        "source1_business_address": s1_row["business_address"], "matched_business_address": m_row["business_address"],
                        "source1_country": s1_row["country"], "matched_country": m_row["country"],
                    })

        rw.subheader("Sampled-pair comparability outcome")
        rw.line(f"MEASURED STATISTIC - total sampled pairs (S1->S2 + S1->S3): {n_sampled_pairs_total}")
        rw.line(f"MEASURED STATISTIC - sampled pairs actually compared (both sides resolved): {n_compared}")
        rw.line(f"MEASURED STATISTIC - sampled pairs NOT compared (one or both sides unresolved): {n_sampled_pairs_total - n_compared}")

        for k, v in sorted(name_cmp_agg.items()):
            pct = round(100.0 * v / n_compared, 4) if n_compared else None
            rw.line(f"MEASURED STATISTIC - {k}: {v} ({pct}%)")
        for k, v in sorted(addr_cmp_agg.items()):
            pct = round(100.0 * v / n_compared, 4) if n_compared else None
            rw.line(f"MEASURED STATISTIC - {k}: {v} ({pct}%)")
        for k, v in sorted(country_cmp_agg.items()):
            pct = round(100.0 * v / n_compared, 4) if n_compared else None
            rw.line(f"MEASURED STATISTIC - {k}: {v} ({pct}%)")

        rw.header("12. MULTILINGUAL TRUE-MATCH ANALYSIS", level=2)
        for cat, count in sorted(multilingual_match_categories.items(), key=lambda kv: -kv[1]):
            rw.line(f"MEASURED STATISTIC - {cat}: {count} sampled true-match pairs")

        rw.header("13. REPRESENTATIVE OBSERVATIONS", level=2)
        for cat, examples in example_rows_by_category.items():
            rw.line(f"REPRESENTATIVE EXAMPLE category: {cat}")
            for ex in examples[:5]:
                rw.line(f"    S1={ex['source1_entity_id']} -> {ex['matched_entity_id']} ({ex['matched_source']}): "
                        f"name={ex['source1_business_name']!r} vs {ex['matched_business_name']!r}")
    else:
        rw.line("Ground truth could not be profiled; true-match analysis skipped.")

    print("Writing representative multilingual samples ...")
    multilingual_rows = []
    for key, profile in file_profiles.items():
        for (field, script), sampler in profile.multilingual_samplers.items():
            for item in sampler.items:
                multilingual_rows.append(item)
    if multilingual_rows:
        pd.DataFrame(multilingual_rows)[["source_file", "field", "script", "entity_id", "value"]].to_csv(
            multilingual_samples_path, sep="\t", index=False, encoding="utf-8"
        )
    else:
        pd.DataFrame(columns=["source_file", "field", "script", "entity_id", "value"]).to_csv(
            multilingual_samples_path, sep="\t", index=False, encoding="utf-8"
        )

    print("Writing representative true-match pairs ...")
    tsv_columns = ["source1_entity_id", "matched_entity_id", "matched_source", "source1_business_name",
                   "matched_business_name", "source1_business_address", "matched_business_address",
                   "source1_country", "matched_country"]
    if true_match_rows_for_tsv:
        pd.DataFrame(true_match_rows_for_tsv)[tsv_columns].to_csv(true_match_samples_path, sep="\t", index=False, encoding="utf-8")
    else:
        pd.DataFrame(columns=tsv_columns).to_csv(true_match_samples_path, sep="\t", index=False, encoding="utf-8")

    rw.header("14. PROCESSING COMPLETENESS / VALIDATION", level=2)
    for key, profile in file_profiles.items():
        rw.line(f"{key}: complete={'YES' if profile.file_complete else 'NO'}, rows={profile.total_rows}, chunks={profile.chunks_processed}, error={profile.error}")
    rw.line(f"train_ground_truth: complete={'YES' if gt_ok else 'NO'}, rows={gt.total_rows}, chunks={gt.chunks_processed}, error={gt.error}")
    if resolution_error:
        rw.line(f"true_match_id_resolution: FAILED, error={resolution_error}")

    multilingual_exists_and_nonempty = os.path.exists(multilingual_samples_path) and os.path.getsize(multilingual_samples_path) > 0
    true_match_exists_and_nonempty = os.path.exists(true_match_samples_path) and os.path.getsize(true_match_samples_path) > 0
    rw.line(f"Output file check - multilingual samples TSV non-empty: {multilingual_exists_and_nonempty}")
    rw.line(f"Output file check - true-match pairs TSV non-empty: {true_match_exists_and_nonempty}")

    overall_ok = all_files_ok and gt_ok and (resolution_error is None)

    print("Writing JSON summary ...")
    for key, profile in file_profiles.items():
        json_summary["files"][key] = profile.to_dict()
    json_summary["ground_truth"] = {
        "total_rows": gt.total_rows,
        "chunks_processed": gt.chunks_processed,
        "unique_source1_entity_id_tracked": gt.source1_id_tracker.distinct_tracked(),
        "duplicate_source1_entity_id": gt.source1_id_tracker.duplicate_count(),
        "source1_id_tracking_capped": gt.source1_id_tracker.capped,
        "zero_match_count": gt.zero_match_count,
        "one_match_count": gt.one_match_count, "multi_match_count": gt.multi_match_count,
        "max_matches": gt.max_matches, "total_matched_id_occurrences": gt.total_matched_id_occurrences,
        "s2_matched_occurrences": gt.s2_matched_occurrences, "s3_matched_occurrences": gt.s3_matched_occurrences,
        "invalid_prefix_occurrences": gt.invalid_prefix_occurrences,
        "malformed_row_count": gt.malformed_row_count,
        "match_count_distribution": dict(gt.match_count_distribution),
        "error": gt.error,
    }
    json_summary["true_match_sampling"] = {
        "s2_sample_size": len(gt.s2_sampler.items) if gt_ok else 0,
        "s3_sample_size": len(gt.s3_sampler.items) if gt_ok else 0,
        "seed": RANDOM_SEED,
        "resolution_error": resolution_error,
    }
    json_summary["processing_status"] = {
        "all_files_complete": all_files_ok, "ground_truth_complete": gt_ok,
        "true_match_resolution_ok": resolution_error is None, "overall_ok": overall_ok,
    }

    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(json_summary, jf, indent=2, ensure_ascii=False, default=str)

    rw.close()

    print()
    print("=" * 60)
    print("PROFILE COMPLETE" if overall_ok else "PROFILE FAILED")
    print("=" * 60)
    for key, profile in file_profiles.items():
        print(f"{key}: rows={profile.total_rows}, status={'OK' if profile.file_complete and not profile.error else 'FAILED'}")
    print(f"train_ground_truth: rows={gt.total_rows}, status={'OK' if gt_ok else 'FAILED'}")
    if resolution_error:
        print(f"true_match_id_resolution: FAILED - {resolution_error}")

    if not overall_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()