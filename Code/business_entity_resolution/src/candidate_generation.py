"""
candidate_generation.py

STEP 4 - High-Recall Candidate Generation / Blocking for the
Amazon ML Challenge 2026 Business Entity Resolution project.

SCOPE (STRICT)
--------------
This module performs ONLY candidate retrieval (blocking) and its
validation against ground truth. It does NOT train a matching model,
compute pairwise ML features, threshold or predict final matches, or
write matching_results.tsv. Those belong to later stages.

ARRAY-BACKED HIT ACCUMULATOR
----------------------------
Per-query hit accumulation for token_name, rare_token_name, char_ngram,
token_address, numeric_address, char_ngram_address, translit_token_name,
and translit_char_ngram uses a fixed-width array("I")-backed
BoundedHitCounter, sized ONCE per target population and stored on
TargetIndexes as `hit_counter`.

The counter:
- Uses array("I") and verifies itemsize == 4 at runtime.
- Is allocated once per active target source (S2 or S3).
- Is reused across all methods and all queries against that target.
- Tracks touched row IDs without a fixed candidate-key cap.
- Resets only touched rows, not the entire target population.
- Never drops a candidate row ID due to an accumulator limit.
- Applies country filtering before hit_counter.add().
- Is reset in try/finally blocks after every accumulation method.

BUILD ARCHITECTURE
------------------
Build modes use a disk-backed, batched, two-pass architecture:
1. Build Source-2 index, stream bounded Source-1 batches, write S2 partial
   candidate rows to an intermediate JSONL file, then free S2.
2. Build Source-3 index, re-stream bounded Source-1 batches, write S3
   partial candidate rows to a second intermediate JSONL file, then free S3.
3. Merge both intermediate files line-by-line, union candidate provenance,
   apply the optional final safety cap, and write final output incrementally.

At no point are all Source-1 rows, S2 indexes, and S3 indexes retained
simultaneously.

TRANSLITERATION POLICY
----------------------
translit_char_ngram is disabled by DEFAULT_METHODS because constructing a
full transliterated character n-gram index across millions of target rows is
expensive. Its implementation remains available and can be enabled using:
    --methods all
or:
    --methods exact_name,token_name,translit_char_ngram

No external data, APIs, geocoding, matching model, thresholding, or final
prediction logic is used.
"""

from __future__ import annotations

import os
import re
import sys
import gc
import dis
import json
import time
import heapq
import types
import random
import logging
import argparse
import unicodedata
import tempfile
from array import array
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Set, Iterable, FrozenSet

import pandas as pd

try:
    from unidecode import unidecode as _unidecode_fn
    TRANSLITERATION_AVAILABLE = True
except ImportError:
    _unidecode_fn = None
    TRANSLITERATION_AVAILABLE = False


logger = logging.getLogger("candidate_generation")
if not logger.handlers:
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
DATASET_TEST_DIR = PROJECT_ROOT / "dataset" / "test"
OUTPUT_DIR = PROJECT_ROOT / "output"

TRAIN_SOURCE_FILES = {
    "source1": DATASET_TRAIN_DIR / "train_source1.tsv",
    "source2": DATASET_TRAIN_DIR / "train_source2.tsv",
    "source3": DATASET_TRAIN_DIR / "train_source3.tsv",
}

TEST_SOURCE_FILES = {
    "source1": DATASET_TEST_DIR / "test_source1.tsv",
    "source2": DATASET_TEST_DIR / "test_source2.tsv",
    "source3": DATASET_TEST_DIR / "test_source3.tsv",
}

GROUND_TRUTH_FILE = DATASET_TRAIN_DIR / "train_ground_truth.tsv"

REQUIRED_SOURCE_COLUMNS = [
    "entity_id",
    "business_name",
    "business_address",
    "country",
]

REQUIRED_GT_COLUMNS = [
    "source1_entity_id",
    "matched_entity_ids",
]

DEFAULT_CHUNKSIZE = 200_000
DEFAULT_BATCH_SIZE = 50_000
RANDOM_SEED = 42

ALL_METHODS = [
    "exact_name",
    "token_name",
    "rare_token_name",
    "token_signature_name",
    "char_ngram",
    "exact_address",
    "token_address",
    "numeric_address",
    "char_ngram_address",
    "translit_exact_name",
    "translit_token_name",
    "translit_char_ngram",
]

DEFAULT_METHODS = [
    method for method in ALL_METHODS
    if method != "translit_char_ngram"
]

METHOD_INDEX_DEPENDENCIES = {
    "exact_name": {"exact_name_index"},
    "token_name": {"token_index", "token_freq"},
    "rare_token_name": {"rare_token_index", "token_freq"},
    "token_signature_name": {"signature_index"},
    "char_ngram": {"ngram_index"},
    "exact_address": {"exact_address_index"},
    "token_address": {"address_token_index"},
    "numeric_address": {"numeric_index"},
    "char_ngram_address": {"address_ngram_index"},
    "translit_exact_name": {"translit_exact_index"},
    "translit_token_name": {"translit_token_index"},
    "translit_char_ngram": {"translit_ngram_index"},
}


@dataclass
class RetrievalConfig:
    methods: List[str] = field(default_factory=lambda: list(DEFAULT_METHODS))
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    char_ngram_top_k: int = 30
    char_ngram_address_top_k: int = 20
    max_postings_per_gram: int = 4000
    max_postings_per_token: int = 4000
    token_df_ratio_threshold: float = 0.20
    rare_token_df_ratio_threshold: float = 0.02
    rare_token_min_shared: int = 2
    final_safety_cap: int = 0
    enable_country_partitioning: bool = True
    chunksize: int = DEFAULT_CHUNKSIZE
    batch_size: int = DEFAULT_BATCH_SIZE
    sample_size: int = 15_000
    seed: int = RANDOM_SEED
    collect_method_diagnostics: bool = False

    def needed_index_structures(self) -> Set[str]:
        needed: Set[str] = set()

        for method in self.methods:
            needed |= METHOD_INDEX_DEPENDENCIES.get(method, set())

        if self.enable_country_partitioning:
            needed.add("country_ids")

        return needed


_UINT32_ARRAY_PROBE = array("I")

if _UINT32_ARRAY_PROBE.itemsize != 4:
    raise RuntimeError(
        f"array('I') is {_UINT32_ARRAY_PROBE.itemsize} bytes on this platform, "
        "but this implementation requires exactly 4-byte unsigned integer slots."
    )


class BoundedHitCounter:
    """
    Reusable array-backed hit counter for one target population.

    The counter has exactly one uint32 slot per target row_id. The
    `touched` list records only rows incremented during the current
    accumulation pass, allowing O(touched) reset complexity.
    """

    __slots__ = ("counts", "touched", "size")

    def __init__(self, size: int):
        if _UINT32_ARRAY_PROBE.itemsize != 4:
            raise RuntimeError(
                "array('I') itemsize is not 4 bytes on this platform."
            )

        self.size = size
        self.counts = array("I", [0]) * size
        self.touched: List[int] = []

    def add(self, row_id: int, increment: int = 1) -> None:
        if self.counts[row_id] == 0:
            self.touched.append(row_id)

        self.counts[row_id] += increment

    def get(self, row_id: int) -> int:
        return self.counts[row_id]

    def touched_rows(self) -> List[int]:
        return self.touched

    def top_k(self, k: int) -> List[Tuple[int, int]]:
        if k <= 0 or not self.touched:
            return []

        heap: List[Tuple[int, int, int]] = []

        for row_id in self.touched:
            count = self.counts[row_id]
            item = (count, -row_id, row_id)

            if len(heap) < k:
                heapq.heappush(heap, item)
            elif item[:2] > heap[0][:2]:
                heapq.heapreplace(heap, item)

        heap.sort(key=lambda item: (item[0], item[1]), reverse=True)

        return [
            (row_id, count)
            for count, _, row_id in heap
        ]
    
    def reset(self) -> None:
        for row_id in self.touched:
            self.counts[row_id] = 0

        self.touched.clear()

    def __len__(self) -> int:
        return len(self.touched)


def _is_missing(value) -> bool:
    if value is None:
        return True

    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False

    if isinstance(result, bool):
        return result

    return False


def _is_usable_string(value) -> bool:
    return isinstance(value, str)


def _nfkc_and_collapse_whitespace(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)

    chars: List[str] = []

    for ch in text:
        if ch in ("\t", "\n", "\r", "\v", "\f"):
            chars.append(" ")
        elif unicodedata.category(ch) == "Zs":
            chars.append(" ")
        else:
            chars.append(ch)

    normalized = "".join(chars).strip()
    normalized = re.sub(r" {2,}", " ", normalized)

    return normalized


def normalize_text(value: Optional[str]) -> Optional[str]:
    if _is_missing(value):
        return None

    if not _is_usable_string(value):
        return None

    return _nfkc_and_collapse_whitespace(value).casefold()


def _is_word_char(ch: str) -> bool:
    category = unicodedata.category(ch)

    return (
        category[0] == "L"
        or category == "Nd"
        or category in ("Mn", "Mc")
    )


def tokenize(normalized_value: Optional[str]) -> List[str]:
    if normalized_value is None:
        return []

    if not isinstance(normalized_value, str):
        return []

    tokens: List[str] = []
    current: List[str] = []

    for ch in normalized_value:
        if _is_word_char(ch):
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []

    if current:
        tokens.append("".join(current))

    return tokens


def char_ngrams(
    text: Optional[str],
    n_lo: int,
    n_hi: int,
) -> Set[str]:
    if not text:
        return set()

    grams: Set[str] = set()
    text_length = len(text)

    for n in range(n_lo, n_hi + 1):
        if text_length < n:
            continue

        for start in range(text_length - n + 1):
            grams.add(text[start:start + n])

    return grams


_DIGIT_RUN_RE = re.compile(r"\d+")


def extract_numeric_tokens(text: Optional[str]) -> Set[str]:
    if not text:
        return set()

    return set(_DIGIT_RUN_RE.findall(text))


def transliterate(text: Optional[str]) -> Optional[str]:
    if not TRANSLITERATION_AVAILABLE:
        return None

    if not text:
        return None

    if not isinstance(text, str):
        return None

    try:
        transliterated = _unidecode_fn(text)
    except Exception:
        return None

    transliterated = transliterated.strip().casefold()

    return transliterated if transliterated else None


@dataclass
class RowStore:
    entity_ids: List[str] = field(default_factory=list)
    names_normalized: List[Optional[str]] = field(default_factory=list)
    names_raw: List[Optional[str]] = field(default_factory=list)
    addresses_normalized: List[Optional[str]] = field(default_factory=list)
    addresses_raw: List[Optional[str]] = field(default_factory=list)
    countries_normalized: List[Optional[str]] = field(default_factory=list)
    country_ids: array = field(default_factory=lambda: array("i"))
    country_to_id: Dict[str, int] = field(default_factory=dict)
    entity_id_to_row: Dict[str, int] = field(default_factory=dict)

    def add_row(
        self,
        entity_id: str,
        name_raw,
        address_raw,
        country_raw,
    ) -> int:
        row_id = len(self.entity_ids)

        self.entity_ids.append(entity_id)
        self.entity_id_to_row[entity_id] = row_id

        self.names_raw.append(
            name_raw if _is_usable_string(name_raw) else None
        )
        self.addresses_raw.append(
            address_raw if _is_usable_string(address_raw) else None
        )

        self.names_normalized.append(normalize_text(name_raw))
        self.addresses_normalized.append(normalize_text(address_raw))

        country_norm = normalize_text(country_raw)
        self.countries_normalized.append(country_norm)

        if country_norm is None:
            self.country_ids.append(-1)
        else:
            country_id = self.country_to_id.get(country_norm)

            if country_id is None:
                country_id = len(self.country_to_id)
                self.country_to_id[country_norm] = country_id

            self.country_ids.append(country_id)

        return row_id

    def __len__(self) -> int:
        return len(self.entity_ids)


def load_source_into_rowstore(
    path: Path,
    chunksize: int,
    max_rows: Optional[int] = None,
) -> RowStore:
    if not path.exists():
        raise FileNotFoundError(f"Required source file not found: {path}")

    store = RowStore()

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=True,
        chunksize=chunksize,
    )

    schema_checked = False
    last_logged_at = 0

    for chunk in reader:
        if not schema_checked:
            missing_columns = [
                column
                for column in REQUIRED_SOURCE_COLUMNS
                if column not in chunk.columns
            ]

            if missing_columns:
                raise ValueError(
                    f"{path}: missing required columns {missing_columns}; "
                    f"found {list(chunk.columns)}"
                )

            schema_checked = True

        for row in chunk.itertuples(index=False):
            store.add_row(
                entity_id=getattr(row, "entity_id"),
                name_raw=getattr(row, "business_name", None),
                address_raw=getattr(row, "business_address", None),
                country_raw=getattr(row, "country", None),
            )

            if max_rows is not None and len(store) >= max_rows:
                del chunk
                return store

            if len(store) - last_logged_at >= 500_000:
                logger.info(
                    "  ... loaded %d rows from %s so far",
                    len(store),
                    path.name,
                )
                last_logged_at = len(store)

        del chunk
        gc.collect()

    return store


class BoundedInvertedIndex:
    def __init__(self, max_postings_per_key: int):
        self.max_postings_per_key = max_postings_per_key
        self._index: Dict[str, array] = {}
        self.capped_keys: Set[str] = set()
        self.total_keys = 0
        self.total_postings = 0
        self.total_dropped_postings = 0

    def add(self, key: str, row_id: int) -> None:
        postings = self._index.get(key)

        if postings is None:
            postings = array("I")
            self._index[key] = postings
            self.total_keys += 1

        if len(postings) < self.max_postings_per_key:
            postings.append(row_id)
            self.total_postings += 1
        else:
            self.capped_keys.add(key)
            self.total_dropped_postings += 1

    def get(self, key: str) -> array:
        return self._index.get(key, array("I"))

    def is_capped(self, key: str) -> bool:
        return key in self.capped_keys

    def diagnostics(self) -> dict:
        return {
            "total_keys": self.total_keys,
            "total_postings": self.total_postings,
            "capped_key_count": len(self.capped_keys),
            "total_dropped_postings": self.total_dropped_postings,
        }


@dataclass
class TokenFrequencyStats:
    doc_frequency: Counter = field(default_factory=Counter)
    n_docs: int = 0

    def ratio(self, token: str) -> float:
        if self.n_docs == 0:
            return 0.0

        return self.doc_frequency.get(token, 0) / self.n_docs


def compute_token_frequency(store: RowStore) -> TokenFrequencyStats:
    stats = TokenFrequencyStats()

    for name_norm in store.names_normalized:
        if name_norm is None:
            continue

        token_set = set(tokenize(name_norm))

        if not token_set:
            continue

        stats.n_docs += 1

        for token in token_set:
            stats.doc_frequency[token] += 1

    return stats


@dataclass
class TargetIndexes:
    store: RowStore
    exact_name_index: Optional[Dict[str, array]] = None
    exact_address_index: Optional[Dict[str, array]] = None
    token_index: Optional[BoundedInvertedIndex] = None
    rare_token_index: Optional[BoundedInvertedIndex] = None
    signature_index: Optional[Dict[FrozenSet[str], array]] = None
    ngram_index: Optional[BoundedInvertedIndex] = None
    address_token_index: Optional[BoundedInvertedIndex] = None
    address_ngram_index: Optional[BoundedInvertedIndex] = None
    numeric_index: Optional[BoundedInvertedIndex] = None
    translit_exact_index: Optional[Dict[str, array]] = None
    translit_token_index: Optional[BoundedInvertedIndex] = None
    translit_ngram_index: Optional[BoundedInvertedIndex] = None
    token_freq: Optional[TokenFrequencyStats] = None
    built_structures: Set[str] = field(default_factory=set)
    hit_counter: Optional[BoundedHitCounter] = None

    def cap_diagnostics(self) -> dict:
        diagnostics = {}

        for name, index in (
            ("token_index", self.token_index),
            ("rare_token_index", self.rare_token_index),
            ("ngram_index", self.ngram_index),
            ("address_token_index", self.address_token_index),
            ("address_ngram_index", self.address_ngram_index),
            ("numeric_index", self.numeric_index),
            ("translit_token_index", self.translit_token_index),
            ("translit_ngram_index", self.translit_ngram_index),
        ):
            if index is not None:
                diagnostics[name] = index.diagnostics()

        return diagnostics


def _add_to_exact_index(
    index: Dict[str, array],
    key: Optional[str],
    row_id: int,
) -> None:
    if key is None:
        return

    postings = index.get(key)

    if postings is None:
        postings = array("I")
        index[key] = postings

    postings.append(row_id)


def build_target_indexes(
    store: RowStore,
    config: RetrievalConfig,
) -> TargetIndexes:
    needed = config.needed_index_structures()

    indexes = TargetIndexes(
        store=store,
        built_structures=needed,
    )

    if "exact_name_index" in needed:
        indexes.exact_name_index = {}

    if "exact_address_index" in needed:
        indexes.exact_address_index = {}

    if "token_index" in needed:
        indexes.token_index = BoundedInvertedIndex(
            config.max_postings_per_token
        )

    if "rare_token_index" in needed:
        indexes.rare_token_index = BoundedInvertedIndex(
            config.max_postings_per_token
        )

    if "signature_index" in needed:
        indexes.signature_index = {}

    if "ngram_index" in needed:
        indexes.ngram_index = BoundedInvertedIndex(
            config.max_postings_per_gram
        )

    if "address_token_index" in needed:
        indexes.address_token_index = BoundedInvertedIndex(
            config.max_postings_per_token
        )

    if "address_ngram_index" in needed:
        indexes.address_ngram_index = BoundedInvertedIndex(
            config.max_postings_per_gram
        )

    if "numeric_index" in needed:
        indexes.numeric_index = BoundedInvertedIndex(
            config.max_postings_per_token
        )

    if "translit_exact_index" in needed:
        indexes.translit_exact_index = {}

    if "translit_token_index" in needed:
        indexes.translit_token_index = BoundedInvertedIndex(
            config.max_postings_per_token
        )

    if "translit_ngram_index" in needed:
        indexes.translit_ngram_index = BoundedInvertedIndex(
            config.max_postings_per_gram
        )

    if "token_freq" in needed:
        indexes.token_freq = compute_token_frequency(store)

    indexes.hit_counter = BoundedHitCounter(len(store))

    any_translit_index_needed = (
        indexes.translit_exact_index is not None
        or indexes.translit_token_index is not None
        or indexes.translit_ngram_index is not None
    )

    total_rows = len(store)
    last_logged_at = 0

    for row_id in range(total_rows):
        name_norm = store.names_normalized[row_id]
        address_norm = store.addresses_normalized[row_id]

        if indexes.exact_name_index is not None:
            _add_to_exact_index(
                indexes.exact_name_index,
                name_norm,
                row_id,
            )

        if name_norm is not None:
            token_set = (
                set(tokenize(name_norm))
                if (
                    indexes.token_index is not None
                    or indexes.rare_token_index is not None
                    or indexes.signature_index is not None
                )
                else set()
            )

            if indexes.token_index is not None and indexes.token_freq is not None:
                for token in token_set:
                    if indexes.token_freq.ratio(token) <= config.token_df_ratio_threshold:
                        indexes.token_index.add(token, row_id)

            if indexes.rare_token_index is not None and indexes.token_freq is not None:
                for token in token_set:
                    if indexes.token_freq.ratio(token) <= config.rare_token_df_ratio_threshold:
                        indexes.rare_token_index.add(token, row_id)

            if indexes.signature_index is not None and token_set:
                signature = frozenset(sorted(token_set))
                postings = indexes.signature_index.get(signature)

                if postings is None:
                    postings = array("I")
                    indexes.signature_index[signature] = postings

                postings.append(row_id)

            if indexes.ngram_index is not None:
                for gram in char_ngrams(
                    name_norm,
                    config.char_ngram_min,
                    config.char_ngram_max,
                ):
                    indexes.ngram_index.add(gram, row_id)

        if indexes.exact_address_index is not None:
            _add_to_exact_index(
                indexes.exact_address_index,
                address_norm,
                row_id,
            )

        if address_norm is not None:
            if indexes.address_token_index is not None:
                for token in set(tokenize(address_norm)):
                    indexes.address_token_index.add(token, row_id)

            if indexes.address_ngram_index is not None:
                for gram in char_ngrams(
                    address_norm,
                    config.char_ngram_min,
                    config.char_ngram_max,
                ):
                    indexes.address_ngram_index.add(gram, row_id)

            if indexes.numeric_index is not None:
                for numeric_token in extract_numeric_tokens(address_norm):
                    indexes.numeric_index.add(numeric_token, row_id)

        if any_translit_index_needed and TRANSLITERATION_AVAILABLE:
            translit_name = transliterate(store.names_raw[row_id])

            if translit_name is not None:
                if indexes.translit_exact_index is not None:
                    _add_to_exact_index(
                        indexes.translit_exact_index,
                        translit_name,
                        row_id,
                    )

                if indexes.translit_token_index is not None:
                    for token in set(tokenize(translit_name)):
                        indexes.translit_token_index.add(token, row_id)

                if indexes.translit_ngram_index is not None:
                    for gram in char_ngrams(
                        translit_name,
                        config.char_ngram_min,
                        config.char_ngram_max,
                    ):
                        indexes.translit_ngram_index.add(gram, row_id)

        if row_id + 1 - last_logged_at >= 500_000:
            logger.info(
                "  ... indexed %d/%d target rows so far",
                row_id + 1,
                total_rows,
            )
            last_logged_at = row_id + 1

    return indexes


class CandidateUnion:
    __slots__ = ("methods_by_candidate", "score_by_candidate")

    def __init__(self):
        self.methods_by_candidate: Dict[Tuple[str, str], Set[str]] = {}
        self.score_by_candidate: Dict[Tuple[str, str], float] = {}

    def add(
        self,
        source_label: str,
        entity_id: str,
        method: str,
        score: Optional[float] = None,
    ) -> None:
        key = (source_label, entity_id)

        methods = self.methods_by_candidate.get(key)

        if methods is None:
            methods = set()
            self.methods_by_candidate[key] = methods

        methods.add(method)

        if score is not None:
            existing_score = self.score_by_candidate.get(key)

            if existing_score is None or score > existing_score:
                self.score_by_candidate[key] = score

    def items(self):
        for key, methods in self.methods_by_candidate.items():
            yield (
                key[0],
                key[1],
                methods,
                self.score_by_candidate.get(key),
            )

    def entity_ids(self) -> Set[str]:
        return {
            entity_id
            for (_source, entity_id) in self.methods_by_candidate.keys()
        }

    def __len__(self) -> int:
        return len(self.methods_by_candidate)


def retrieve_candidates_for_entity(
    query_name_norm: Optional[str],
    query_name_raw: Optional[str],
    query_addr_norm: Optional[str],
    query_country_id: int,
    idx: TargetIndexes,
    source_label: str,
    config: RetrievalConfig,
    per_method_candidate_ids: Optional[Dict[str, Set[str]]] = None,
    capped_keys_touched: Optional[Set[str]] = None,
) -> CandidateUnion:
    union = CandidateUnion()
    store = idx.store
    hit_counter = idx.hit_counter

    def country_ok(candidate_row_id: int) -> bool:
        if not config.enable_country_partitioning:
            return True

        if query_country_id == -1:
            return True

        candidate_country_id = store.country_ids[candidate_row_id]

        if candidate_country_id == -1:
            return True

        return query_country_id == candidate_country_id

    def emit(
        method: str,
        row_ids: Iterable[int],
        score: Optional[float] = None,
    ) -> None:
        method_candidates = (
            per_method_candidate_ids.setdefault(method, set())
            if per_method_candidate_ids is not None
            else None
        )

        for row_id in row_ids:
            if not country_ok(row_id):
                continue

            entity_id = store.entity_ids[row_id]
            union.add(source_label, entity_id, method, score)

            if method_candidates is not None:
                method_candidates.add(entity_id)

    if query_name_norm is not None:
        if idx.exact_name_index is not None:
            emit(
                "exact_name",
                idx.exact_name_index.get(query_name_norm, ()),
                score=1.0,
            )

        token_set = set(tokenize(query_name_norm))

        if idx.token_index is not None and idx.token_freq is not None:
            try:
                for token in token_set:
                    if idx.token_freq.ratio(token) <= config.token_df_ratio_threshold:
                        if (
                            capped_keys_touched is not None
                            and idx.token_index.is_capped(token)
                        ):
                            capped_keys_touched.add(f"token_name:{token}")

                        for row_id in idx.token_index.get(token):
                            if country_ok(row_id):
                                hit_counter.add(row_id)

                emit(
                    "token_name",
                    hit_counter.touched_rows(),
                )
            finally:
                hit_counter.reset()

        if idx.rare_token_index is not None and idx.token_freq is not None:
            rare_tokens = [
                token
                for token in token_set
                if idx.token_freq.ratio(token)
                <= config.rare_token_df_ratio_threshold
            ]

            try:
                for token in rare_tokens:
                    if (
                        capped_keys_touched is not None
                        and idx.rare_token_index.is_capped(token)
                    ):
                        capped_keys_touched.add(
                            f"rare_token_name:{token}"
                        )

                    for row_id in idx.rare_token_index.get(token):
                        if country_ok(row_id):
                            hit_counter.add(row_id)

                min_shared = min(
                    config.rare_token_min_shared,
                    max(1, len(rare_tokens)),
                )

                qualifying_rows = [
                    row_id
                    for row_id in hit_counter.touched_rows()
                    if hit_counter.get(row_id) >= min_shared
                ]

                emit(
                    "rare_token_name",
                    qualifying_rows,
                )
            finally:
                hit_counter.reset()

        if idx.signature_index is not None and token_set:
            signature = frozenset(sorted(token_set))

            emit(
                "token_signature_name",
                idx.signature_index.get(signature, ()),
                score=1.0,
            )

        if idx.ngram_index is not None:
            grams = char_ngrams(
                query_name_norm,
                config.char_ngram_min,
                config.char_ngram_max,
            )

            try:
                for gram in grams:
                    if (
                        capped_keys_touched is not None
                        and idx.ngram_index.is_capped(gram)
                    ):
                        capped_keys_touched.add(f"char_ngram:{gram}")

                    for row_id in idx.ngram_index.get(gram):
                        if country_ok(row_id):
                            hit_counter.add(row_id)

                top_rows = hit_counter.top_k(
                    config.char_ngram_top_k
                )

                max_possible = max(len(grams), 1)

                for row_id, count in top_rows:
                    entity_id = store.entity_ids[row_id]

                    union.add(
                        source_label,
                        entity_id,
                        "char_ngram",
                        count / max_possible,
                    )

                    if per_method_candidate_ids is not None:
                        per_method_candidate_ids.setdefault(
                            "char_ngram",
                            set(),
                        ).add(entity_id)
            finally:
                hit_counter.reset()

    if query_addr_norm is not None:
        if idx.exact_address_index is not None:
            emit(
                "exact_address",
                idx.exact_address_index.get(query_addr_norm, ()),
                score=1.0,
            )

        address_token_set = set(tokenize(query_addr_norm))

        if idx.address_token_index is not None:
            try:
                for token in address_token_set:
                    if (
                        capped_keys_touched is not None
                        and idx.address_token_index.is_capped(token)
                    ):
                        capped_keys_touched.add(
                            f"token_address:{token}"
                        )

                    for row_id in idx.address_token_index.get(token):
                        if country_ok(row_id):
                            hit_counter.add(row_id)

                emit(
                    "token_address",
                    hit_counter.touched_rows(),
                )
            finally:
                hit_counter.reset()

        if idx.numeric_index is not None:
            try:
                for numeric_token in extract_numeric_tokens(
                    query_addr_norm
                ):
                    for row_id in idx.numeric_index.get(
                        numeric_token
                    ):
                        if country_ok(row_id):
                            hit_counter.add(row_id)

                emit(
                    "numeric_address",
                    hit_counter.touched_rows(),
                )
            finally:
                hit_counter.reset()

        if idx.address_ngram_index is not None:
            grams = char_ngrams(
                query_addr_norm,
                config.char_ngram_min,
                config.char_ngram_max,
            )

            try:
                for gram in grams:
                    if (
                        capped_keys_touched is not None
                        and idx.address_ngram_index.is_capped(gram)
                    ):
                        capped_keys_touched.add(
                            f"char_ngram_address:{gram}"
                        )

                    for row_id in idx.address_ngram_index.get(gram):
                        if country_ok(row_id):
                            hit_counter.add(row_id)

                top_rows = hit_counter.top_k(
                    config.char_ngram_address_top_k
                )

                max_possible = max(len(grams), 1)

                for row_id, count in top_rows:
                    entity_id = store.entity_ids[row_id]

                    union.add(
                        source_label,
                        entity_id,
                        "char_ngram_address",
                        count / max_possible,
                    )

                    if per_method_candidate_ids is not None:
                        per_method_candidate_ids.setdefault(
                            "char_ngram_address",
                            set(),
                        ).add(entity_id)
            finally:
                hit_counter.reset()

    if TRANSLITERATION_AVAILABLE and query_name_raw is not None:
        translit_query = transliterate(query_name_raw)

        if translit_query is not None:
            if idx.translit_exact_index is not None:
                emit(
                    "translit_exact_name",
                    idx.translit_exact_index.get(translit_query, ()),
                    score=1.0,
                )

            if idx.translit_token_index is not None:
                try:
                    for token in set(tokenize(translit_query)):
                        for row_id in idx.translit_token_index.get(token):
                            if country_ok(row_id):
                                hit_counter.add(row_id)

                    emit(
                        "translit_token_name",
                        hit_counter.touched_rows(),
                    )
                finally:
                    hit_counter.reset()

            if idx.translit_ngram_index is not None:
                grams = char_ngrams(
                    translit_query,
                    config.char_ngram_min,
                    config.char_ngram_max,
                )

                try:
                    for gram in grams:
                        for row_id in idx.translit_ngram_index.get(gram):
                            if country_ok(row_id):
                                hit_counter.add(row_id)

                    top_rows = hit_counter.top_k(
                        config.char_ngram_top_k
                    )

                    max_possible = max(len(grams), 1)

                    for row_id, count in top_rows:
                        entity_id = store.entity_ids[row_id]

                        union.add(
                            source_label,
                            entity_id,
                            "translit_char_ngram",
                            count / max_possible,
                        )

                        if per_method_candidate_ids is not None:
                            per_method_candidate_ids.setdefault(
                                "translit_char_ngram",
                                set(),
                            ).add(entity_id)
                finally:
                    hit_counter.reset()

    return union


def apply_final_safety_cap(
    union_items: List[Tuple[str, str, Set[str], Optional[float]]],
    cap: int,
    true_matched_set: Optional[Set[str]] = None,
) -> Tuple[List[Tuple[str, str, Set[str], Optional[float]]], dict]:
    diagnostics = {
        "count_before_cap": len(union_items),
        "cap": cap,
        "truncated": False,
        "removed_candidates_sample": [],
        "true_matches_lost": 0,
        "removed_true_matches_sample": [],
    }

    if cap == 0 or len(union_items) <= cap:
        return union_items, diagnostics

    ordered = sorted(
        union_items,
        key=lambda item: (
            len(item[2]),
            item[3] if item[3] is not None else 0.0,
        ),
        reverse=True,
    )

    kept = ordered[:cap]
    removed = ordered[cap:]

    diagnostics["truncated"] = True

    if true_matched_set:
        removed_ids = {
            entity_id
            for _source, entity_id, _methods, _score in removed
        }

        lost = removed_ids & true_matched_set

        diagnostics["true_matches_lost"] = len(lost)
        diagnostics["removed_true_matches_sample"] = sorted(lost)[:20]

    diagnostics["removed_candidates_sample"] = [
        f"{source}:{entity_id}"
        for source, entity_id, _methods, _score in removed[:50]
    ]

    return kept, diagnostics


@dataclass
class QueryRecord:
    entity_id: str
    name_norm: Optional[str]
    name_raw: Optional[str]
    addr_norm: Optional[str]
    country_norm: Optional[str]


def normalize_query_row(
    entity_id: str,
    name_raw,
    address_raw,
    country_raw,
) -> QueryRecord:
    return QueryRecord(
        entity_id=entity_id,
        name_norm=normalize_text(name_raw),
        name_raw=name_raw if isinstance(name_raw, str) else None,
        addr_norm=normalize_text(address_raw),
        country_norm=normalize_text(country_raw),
    )


def iter_source1_batches(
    path: Path,
    chunksize: int,
    batch_size: int,
    restrict_to: Optional[Set[str]] = None,
) -> Iterable[List[QueryRecord]]:
    if not path.exists():
        raise FileNotFoundError(f"Source-1 file not found: {path}")

    remaining = set(restrict_to) if restrict_to is not None else None
    batch: List[QueryRecord] = []

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=True,
        chunksize=chunksize,
    )

    schema_checked = False

    for chunk in reader:
        if not schema_checked:
            missing = [
                column
                for column in REQUIRED_SOURCE_COLUMNS
                if column not in chunk.columns
            ]

            if missing:
                raise ValueError(
                    f"{path}: missing required columns {missing}"
                )

            schema_checked = True

        for row in chunk.itertuples(index=False):
            entity_id = getattr(row, "entity_id")

            if remaining is not None:
                if entity_id not in remaining:
                    continue

                remaining.discard(entity_id)

            batch.append(
                normalize_query_row(
                    entity_id,
                    getattr(row, "business_name", None),
                    getattr(row, "business_address", None),
                    getattr(row, "country", None),
                )
            )

            if len(batch) >= batch_size:
                yield batch
                batch = []

            if remaining is not None and not remaining:
                break

        del chunk
        gc.collect()

        if remaining is not None and not remaining:
            break

    if batch:
        yield batch


def resolve_country_id(
    country_norm: Optional[str],
    target_country_to_id: Dict[str, int],
) -> int:
    if country_norm is None:
        return -1

    return target_country_to_id.get(country_norm, -1)


def _write_partial_jsonl(
    fh,
    entity_id: str,
    union: CandidateUnion,
) -> None:
    payload = {
        "s1": entity_id,
        "c": [
            [source, candidate_id, sorted(methods), score]
            for source, candidate_id, methods, score in union.items()
        ],
    }

    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _pass_over_source1_against_target(
    s1_path: Path,
    target_path: Path,
    source_label: str,
    config: RetrievalConfig,
    out_jsonl_path: Path,
    restrict_to: Optional[Set[str]],
    per_query_method_cache: Optional[
        Dict[str, Dict[str, Set[str]]]
    ] = None,
    capped_keys_by_query: Optional[Dict[str, Set[str]]] = None,
) -> dict:
    t0 = time.time()

    store = load_source_into_rowstore(
        target_path,
        config.chunksize,
    )

    t_loaded = time.time()

    indexes = build_target_indexes(store, config)

    t_indexed = time.time()

    n_rows_written = 0
    n_batches = 0

    with open(out_jsonl_path, "w", encoding="utf-8") as out_fh:
        for batch in iter_source1_batches(
            s1_path,
            config.chunksize,
            config.batch_size,
            restrict_to=restrict_to,
        ):
            n_batches += 1

            for query in batch:
                country_id = resolve_country_id(
                    query.country_norm,
                    store.country_to_id,
                )

                method_cache = (
                    per_query_method_cache.setdefault(
                        query.entity_id,
                        {},
                    )
                    if per_query_method_cache is not None
                    else None
                )

                capped_set = (
                    capped_keys_by_query.setdefault(
                        query.entity_id,
                        set(),
                    )
                    if capped_keys_by_query is not None
                    else None
                )

                union = retrieve_candidates_for_entity(
                    query.name_norm,
                    query.name_raw,
                    query.addr_norm,
                    country_id,
                    indexes,
                    source_label,
                    config,
                    per_method_candidate_ids=method_cache,
                    capped_keys_touched=capped_set,
                )

                _write_partial_jsonl(
                    out_fh,
                    query.entity_id,
                    union,
                )

                n_rows_written += 1

            del batch
            gc.collect()

            if n_batches % 5 == 0:
                logger.info(
                    "  ... %s pass: %d batches processed (%d Source-1 rows so far)",
                    source_label,
                    n_batches,
                    n_rows_written,
                )

    diagnostics = {
        "rows_loaded": len(store),
        "load_seconds": round(t_loaded - t0, 2),
        "index_build_seconds": round(t_indexed - t_loaded, 2),
        "query_and_write_seconds": round(
            time.time() - t_indexed,
            2,
        ),
        "structures_built": sorted(indexes.built_structures),
        "n_batches": n_batches,
        "n_query_rows_written": n_rows_written,
        "index_cap_diagnostics": indexes.cap_diagnostics(),
    }

    del store
    del indexes
    gc.collect()

    return diagnostics


def _merge_partial_files_and_write_final(
    s2_jsonl_path: Path,
    s3_jsonl_path: Path,
    config: RetrievalConfig,
    writer: "CandidatePairWriter",
) -> dict:
    n_rows = 0
    n_with_candidates = 0
    n_truncated = 0
    total_candidates = 0

    with open(s2_jsonl_path, "r", encoding="utf-8") as f2:
        with open(s3_jsonl_path, "r", encoding="utf-8") as f3:
            for line2, line3 in zip(f2, f3):
                rec2 = json.loads(line2)
                rec3 = json.loads(line3)

                if rec2["s1"] != rec3["s1"]:
                    raise RuntimeError(
                        "Intermediate file order mismatch: "
                        f"{rec2['s1']} vs {rec3['s1']}"
                    )

                entity_id = rec2["s1"]

                combined_items: List[
                    Tuple[str, str, Set[str], Optional[float]]
                ] = []

                for source, candidate_id, methods, score in (
                    rec2["c"] + rec3["c"]
                ):
                    combined_items.append(
                        (
                            source,
                            candidate_id,
                            set(methods),
                            score,
                        )
                    )

                dedup: Dict[
                    Tuple[str, str],
                    Tuple[Set[str], Optional[float]],
                ] = {}

                for source, candidate_id, methods, score in combined_items:
                    key = (source, candidate_id)

                    if key in dedup:
                        existing_methods, existing_score = dedup[key]
                        existing_methods |= methods

                        if (
                            score is not None
                            and (
                                existing_score is None
                                or score > existing_score
                            )
                        ):
                            dedup[key] = (
                                existing_methods,
                                score,
                            )
                    else:
                        dedup[key] = (
                            methods,
                            score,
                        )

                final_items = [
                    (source, candidate_id, methods, score)
                    for (source, candidate_id), (methods, score)
                    in dedup.items()
                ]

                kept, truncation_diag = apply_final_safety_cap(
                    final_items,
                    config.final_safety_cap,
                )

                writer.write(entity_id, kept)

                n_rows += 1

                if kept:
                    n_with_candidates += 1

                if truncation_diag["truncated"]:
                    n_truncated += 1

                total_candidates += len(kept)

            leftover2 = f2.readline()
            leftover3 = f3.readline()

            if leftover2 or leftover3:
                raise RuntimeError(
                    "Intermediate files have mismatched lengths"
                )

    return {
        "n_rows": n_rows,
        "n_with_candidates": n_with_candidates,
        "n_truncated": n_truncated,
        "avg_candidates_per_row": (
            total_candidates / n_rows
            if n_rows
            else 0.0
        ),
    }


def run_build_pipeline(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    config: RetrievalConfig,
    writer: "CandidatePairWriter",
) -> dict:
    with tempfile.TemporaryDirectory(prefix="candgen_") as tmp_dir:
        s2_partial_path = Path(tmp_dir) / "s2_partial.jsonl"
        s3_partial_path = Path(tmp_dir) / "s3_partial.jsonl"

        logger.info(
            "PASS 1/3: building Source-2 index and streaming Source-1 "
            "against it (batch_size=%d) ...",
            config.batch_size,
        )

        s2_diag = _pass_over_source1_against_target(
            s1_path,
            s2_path,
            "S2",
            config,
            s2_partial_path,
            restrict_to=None,
        )

        logger.info("PASS 1/3 complete: %s", s2_diag)

        logger.info(
            "PASS 2/3: building Source-3 index and re-streaming "
            "Source-1 against it ..."
        )

        s3_diag = _pass_over_source1_against_target(
            s1_path,
            s3_path,
            "S3",
            config,
            s3_partial_path,
            restrict_to=None,
        )

        logger.info("PASS 2/3 complete: %s", s3_diag)

        logger.info(
            "PASS 3/3: merging S2-side and S3-side partial candidates "
            "(synchronized line-by-line) and writing final output ..."
        )

        merge_summary = _merge_partial_files_and_write_final(
            s2_partial_path,
            s3_partial_path,
            config,
            writer,
        )

        logger.info("PASS 3/3 complete: %s", merge_summary)

    return {
        "pass1_source2": s2_diag,
        "pass2_source3": s3_diag,
        "pass3_merge": merge_summary,
    }


def parse_matched_ids(raw_value) -> List[str]:
    if _is_missing(raw_value):
        return []

    if not isinstance(raw_value, str):
        raw_value = str(raw_value)

    if raw_value.strip() == "":
        return []

    return [
        token
        for token in raw_value.split(",")
        if token != ""
    ]


def load_ground_truth_sample(
    path: Path,
    sample_size: Optional[int],
    seed: int,
) -> Dict[str, List[str]]:
    """
    TWO-PASS memory-safe ground-truth sampler:
    - Sample mode: reservoir-sample S1 IDs in pass 1, fetch only sampled
      matched lists in pass 2.
    - Full mode: sample_size=None builds the complete streamed map.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Ground truth file not found: {path}"
        )

    if sample_size is None:
        full_map: Dict[str, List[str]] = {}

        reader = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=True,
            chunksize=DEFAULT_CHUNKSIZE,
        )

        schema_checked = False

        for chunk in reader:
            if not schema_checked:
                missing = [
                    column
                    for column in REQUIRED_GT_COLUMNS
                    if column not in chunk.columns
                ]

                if missing:
                    raise ValueError(
                        f"Ground truth missing columns {missing}; "
                        f"found {list(chunk.columns)}"
                    )

                schema_checked = True

            for s1_id, raw_matched in zip(
                chunk["source1_entity_id"],
                chunk["matched_entity_ids"],
            ):
                if _is_missing(s1_id):
                    continue

                full_map[str(s1_id)] = parse_matched_ids(
                    raw_matched
                )

            del chunk
            gc.collect()

        return full_map

    rng = random.Random(seed)
    reservoir: List[str] = []
    n_seen = 0

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=True,
        chunksize=DEFAULT_CHUNKSIZE,
    )

    schema_checked = False

    for chunk in reader:
        if not schema_checked:
            missing = [
                column
                for column in REQUIRED_GT_COLUMNS
                if column not in chunk.columns
            ]

            if missing:
                raise ValueError(
                    f"Ground truth missing columns {missing}; "
                    f"found {list(chunk.columns)}"
                )

            schema_checked = True

        for s1_id in chunk["source1_entity_id"]:
            if _is_missing(s1_id):
                continue

            n_seen += 1
            s1_id = str(s1_id)

            if len(reservoir) < sample_size:
                reservoir.append(s1_id)
            else:
                j = rng.randint(0, n_seen - 1)

                if j < sample_size:
                    reservoir[j] = s1_id

        del chunk
        gc.collect()

    sample_ids = set(reservoir)
    del reservoir

    logger.info(
        "GT sampling Pass 1: reservoir-sampled %d ids out of %d seen "
        "(target sample_size=%d)",
        len(sample_ids),
        n_seen,
        sample_size,
    )

    gt_map: Dict[str, List[str]] = {}
    remaining = set(sample_ids)

    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=True,
        chunksize=DEFAULT_CHUNKSIZE,
    )

    for chunk in reader:
        hit_mask = chunk["source1_entity_id"].isin(remaining)

        if hit_mask.any():
            hits = chunk.loc[hit_mask]

            for s1_id, raw_matched in zip(
                hits["source1_entity_id"],
                hits["matched_entity_ids"],
            ):
                s1_id = str(s1_id)
                gt_map[s1_id] = parse_matched_ids(raw_matched)
                remaining.discard(s1_id)

        del chunk

        if not remaining:
            break

        gc.collect()

    logger.info(
        "GT sampling Pass 2: resolved matched_entity_ids for %d/%d "
        "sampled ids",
        len(gt_map),
        len(sample_ids),
    )

    return gt_map


@dataclass
class RecallResult:
    n_queries: int = 0
    n_zero_match_queries: int = 0
    n_with_truth: int = 0
    n_zero_candidate_queries: int = 0
    true_matches_total: int = 0
    true_matches_recovered: int = 0
    true_matches_recovered_s2: int = 0
    true_matches_total_s2: int = 0
    true_matches_recovered_s3: int = 0
    true_matches_total_s3: int = 0
    candidate_counts: List[int] = field(default_factory=list)
    zero_match_candidate_counts: List[int] = field(default_factory=list)
    zero_match_with_any_candidate: int = 0
    per_cardinality_recovered: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    per_cardinality_total: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    per_cardinality_queries: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    difficult_case_recovered: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    difficult_case_total: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    country_recovered: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    country_total: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int)
    )
    truncation_events: int = 0
    truncation_lost_true_matches: int = 0
    candidate_counts_before_cap: List[int] = field(default_factory=list)
    cap_possibly_affected_true_matches: int = 0

    def overall_recall(self) -> Optional[float]:
        if self.true_matches_total == 0:
            return None

        return self.true_matches_recovered / self.true_matches_total

    def recall_s2(self) -> Optional[float]:
        if self.true_matches_total_s2 == 0:
            return None

        return self.true_matches_recovered_s2 / self.true_matches_total_s2

    def recall_s3(self) -> Optional[float]:
        if self.true_matches_total_s3 == 0:
            return None

        return self.true_matches_recovered_s3 / self.true_matches_total_s3


def _cardinality_bucket(n_true: int) -> str:
    if n_true == 0:
        return "0"

    if n_true == 1:
        return "1"

    return "2+"


def _percentile(
    sorted_values: List[int],
    percentile: float,
) -> Optional[int]:
    if not sorted_values:
        return None

    index = min(
        int(percentile / 100.0 * len(sorted_values)),
        len(sorted_values) - 1,
    )

    return sorted_values[index]


_SCRIPT_RANGES_FOR_DIFFICULTY = {
    "Latin": [(0x0041, 0x024F), (0x1E00, 0x1EFF)],
    "Devanagari": [(0x0900, 0x097F)],
    "Gurmukhi": [(0x0A00, 0x0A7F)],
    "Bengali": [(0x0980, 0x09FF)],
    "Gujarati": [(0x0A80, 0x0AFF)],
    "Tamil": [(0x0B80, 0x0BFF)],
    "Telugu": [(0x0C00, 0x0C7F)],
    "Kannada": [(0x0C80, 0x0CFF)],
    "Malayalam": [(0x0D00, 0x0D7F)],
    "Odia": [(0x0B00, 0x0B7F)],
    "Arabic": [(0x0600, 0x06FF)],
    "Hebrew": [(0x0590, 0x05FF)],
    "Cyrillic": [(0x0400, 0x04FF)],
    "Han": [(0x4E00, 0x9FFF)],
}


def _scripts_in(text: Optional[str]) -> Set[str]:
    if not text:
        return set()

    found: Set[str] = set()

    for ch in text:
        codepoint = ord(ch)

        for script, ranges in _SCRIPT_RANGES_FOR_DIFFICULTY.items():
            for lower, upper in ranges:
                if lower <= codepoint <= upper:
                    found.add(script)

    return found


def run_validation_pipeline(
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    gt_map: Dict[str, List[str]],
    config: RetrievalConfig,
) -> Tuple[RecallResult, dict]:
    diagnostics: Dict[str, dict] = {}
    query_ids = set(gt_map.keys())

    queries: List[QueryRecord] = []

    for batch in iter_source1_batches(
        s1_path,
        config.chunksize,
        config.batch_size,
        restrict_to=set(query_ids),
    ):
        queries.extend(batch)

    diagnostics["stage_load_queries"] = {
        "n_queries_loaded": len(queries),
        "n_queries_requested": len(query_ids),
    }

    logger.info(
        "Loaded %d query records (requested %d)",
        len(queries),
        len(query_ids),
    )

    query_by_id = {q.entity_id: q for q in queries}
    capped_keys_by_query: Dict[str, Set[str]] = defaultdict(set)
    index_cap_diag: Dict[str, dict] = {}

    with tempfile.TemporaryDirectory(prefix="candgen_val_") as tmp_dir:
        s2_partial_path = Path(tmp_dir) / "s2_partial.jsonl"
        s3_partial_path = Path(tmp_dir) / "s3_partial.jsonl"

        for source_label, source_path, partial_path in (
            ("S2", s2_path, s2_partial_path),
            ("S3", s3_path, s3_partial_path),
        ):
            stage_start = time.time()

            logger.info("Loading and indexing %s ...", source_label)

            store = load_source_into_rowstore(
                source_path,
                config.chunksize,
            )

            loaded_at = time.time()

            logger.info(
                "%s loaded: %d rows in %.2fs. Building indexes for active methods ...",
                source_label,
                len(store),
                loaded_at - stage_start,
            )

            indexes = build_target_indexes(store, config)

            indexed_at = time.time()

            logger.info(
                "%s indexes built in %.2fs (structures: %s). Querying %d entities ...",
                source_label,
                indexed_at - loaded_at,
                sorted(indexes.built_structures),
                len(queries),
            )

            n_done = 0

            with open(partial_path, "w", encoding="utf-8") as out_fh:
                for query in queries:
                    query_country_id = resolve_country_id(
                        query.country_norm,
                        store.country_to_id,
                    )

                    capped_set = capped_keys_by_query[query.entity_id]

                    sub_union = retrieve_candidates_for_entity(
                        query.name_norm,
                        query.name_raw,
                        query.addr_norm,
                        query_country_id,
                        indexes,
                        source_label,
                        config,
                        per_method_candidate_ids=None,
                        capped_keys_touched=capped_set,
                    )

                    _write_partial_jsonl(out_fh, query.entity_id, sub_union)

                    n_done += 1

                    if n_done % 2500 == 0:
                        logger.info(
                            "  ... %s: queried %d/%d entities",
                            source_label,
                            n_done,
                            len(queries),
                        )

            queried_at = time.time()

            index_cap_diag[source_label] = indexes.cap_diagnostics()

            diagnostics[f"stage_{source_label.lower()}"] = {
                "rows_loaded": len(store),
                "load_seconds": round(loaded_at - stage_start, 2),
                "index_build_seconds": round(indexed_at - loaded_at, 2),
                "query_seconds": round(queried_at - indexed_at, 2),
                "structures_built": sorted(indexes.built_structures),
            }

            logger.info(
                "%s: %d rows loaded in %.2fs, indexed in %.2fs, queried in %.2fs",
                source_label,
                len(store),
                loaded_at - stage_start,
                indexed_at - loaded_at,
                queried_at - indexed_at,
            )

            del store
            del indexes
            gc.collect()

        diagnostics["posting_cap_diagnostics"] = index_cap_diag

        result = RecallResult()

        method_contribution: Dict[str, dict] = {}
        if config.collect_method_diagnostics:
            method_contribution = {
                method: {
                    "candidates_contributed": 0,
                    "true_matches_recovered": 0,
                    "true_matches_uniquely_recovered": 0,
                }
                for method in config.methods
            }

        ablation_groups = {
            "exact_name_only": ["exact_name"],
            "exact_plus_token": ["exact_name", "token_name"],
            "name_methods": [
                "exact_name",
                "token_name",
                "rare_token_name",
                "token_signature_name",
                "char_ngram",
            ],
            "address_methods": [
                "exact_address",
                "token_address",
                "numeric_address",
                "char_ngram_address",
            ],
            "name_plus_address": [
                "exact_name",
                "token_name",
                "rare_token_name",
                "token_signature_name",
                "char_ngram",
                "exact_address",
                "token_address",
                "numeric_address",
                "char_ngram_address",
            ],
            "all_non_translit": [
                method
                for method in ALL_METHODS
                if not method.startswith("translit_")
            ],
            "all_enabled": list(config.methods),
        }

        if TRANSLITERATION_AVAILABLE and any(method.startswith("translit_") for method in config.methods):
            ablation_groups["translit_only"] = [
                method
                for method in config.methods
                if method.startswith("translit_")
            ]

        ablation_totals = {g: {"true": 0, "recovered": 0} for g in ablation_groups} if config.collect_method_diagnostics else {}

        with open(s2_partial_path, "r", encoding="utf-8") as f2:
            with open(s3_partial_path, "r", encoding="utf-8") as f3:
                for line2, line3 in zip(f2, f3):
                    rec2 = json.loads(line2)
                    rec3 = json.loads(line3)

                    if rec2["s1"] != rec3["s1"]:
                        raise RuntimeError(f"Intermediate file order mismatch: {rec2['s1']} vs {rec3['s1']}")

                    entity_id = rec2["s1"]
                    query = query_by_id[entity_id]
                    
                    true_matches = gt_map.get(entity_id, [])
                    n_true = len(true_matches)
                    true_set = set(true_matches)

                    true_s2 = [match_id for match_id in true_matches if match_id.startswith("S2-")]
                    true_s3 = [match_id for match_id in true_matches if match_id.startswith("S3-")]

                    combined_items: List[Tuple[str, str, Set[str], Optional[float]]] = []
                    for source, candidate_id, methods, score in (rec2["c"] + rec3["c"]):
                        combined_items.append((source, candidate_id, set(methods), score))

                    dedup: Dict[Tuple[str, str], Tuple[Set[str], Optional[float]]] = {}
                    for source, candidate_id, methods, score in combined_items:
                        key = (source, candidate_id)
                        if key in dedup:
                            existing_methods, existing_score = dedup[key]
                            existing_methods |= methods
                            if score is not None and (existing_score is None or score > existing_score):
                                dedup[key] = (existing_methods, score)
                        else:
                            dedup[key] = (methods, score)

                    final_items = [
                        (source, candidate_id, methods, score)
                        for (source, candidate_id), (methods, score) in dedup.items()
                    ]

                    result.candidate_counts_before_cap.append(len(final_items))

                    kept, truncation_diag = apply_final_safety_cap(
                        final_items,
                        config.final_safety_cap,
                        true_matched_set=true_set,
                    )

                    candidate_ids = {
                        candidate_id
                        for _source, candidate_id, _methods, _score in kept
                    }

                    if not candidate_ids:
                        result.n_zero_candidate_queries += 1

                    if truncation_diag["truncated"]:
                        result.truncation_events += 1
                        result.truncation_lost_true_matches += truncation_diag["true_matches_lost"]

                    result.n_queries += 1
                    cardinality_bucket = _cardinality_bucket(n_true)
                    result.per_cardinality_queries[cardinality_bucket] += 1
                    query_country = query.country_norm or "__missing__"
                    capped_keys_for_query = capped_keys_by_query.get(entity_id, set())

                    if n_true > 0:
                        missed = true_set - candidate_ids
                        if missed and capped_keys_for_query:
                            result.cap_possibly_affected_true_matches += len(missed)

                    if config.collect_method_diagnostics:
                        query_method_candidates = defaultdict(set)
                        for source, candidate_id, methods, score in final_items:
                            for method in methods:
                                query_method_candidates[method].add(candidate_id)
                        
                        for method in config.methods:
                            method_ids = query_method_candidates.get(method, set())
                            method_contribution[method]["candidates_contributed"] += len(method_ids)
                            method_contribution[method]["true_matches_recovered"] += len(method_ids & true_set)

                    if n_true == 0:
                        result.n_zero_match_queries += 1
                        result.zero_match_candidate_counts.append(len(candidate_ids))
                        if candidate_ids:
                            result.zero_match_with_any_candidate += 1
                        continue

                    result.n_with_truth += 1
                    recovered = candidate_ids & true_set

                    result.true_matches_total += n_true
                    result.true_matches_recovered += len(recovered)
                    result.true_matches_total_s2 += len(true_s2)
                    result.true_matches_recovered_s2 += len(candidate_ids & set(true_s2))
                    result.true_matches_total_s3 += len(true_s3)
                    result.true_matches_recovered_s3 += len(candidate_ids & set(true_s3))
                    result.candidate_counts.append(len(candidate_ids))
                    result.per_cardinality_total[cardinality_bucket] += n_true
                    result.per_cardinality_recovered[cardinality_bucket] += len(recovered)
                    result.country_total[query_country] += n_true
                    result.country_recovered[query_country] += len(recovered)

                    name_scripts = _scripts_in(query.name_norm)
                    difficulty_tags = [
                        ("multilingual_name" if name_scripts - {"Latin"} else "latin_name"),
                        ("missing_address" if query.addr_norm is None else "address_present"),
                    ]

                    for tag in difficulty_tags:
                        result.difficult_case_total[tag] += n_true
                        result.difficult_case_recovered[tag] += len(recovered)

                    if config.collect_method_diagnostics:
                        for method in config.methods:
                            all_method_ids: Set[str] = set()
                            without_method_ids: Set[str] = set()
                            for other_method in config.methods:
                                ids = query_method_candidates.get(other_method, set())
                                all_method_ids |= ids
                                if other_method != method:
                                    without_method_ids |= ids
                            uniquely_recovered = (all_method_ids & true_set) - (without_method_ids & true_set)
                            method_contribution[method]["true_matches_uniquely_recovered"] += len(uniquely_recovered)

                        for group_name, group_methods in ablation_groups.items():
                            active_group_methods = [m for m in group_methods if m in config.methods]
                            if not active_group_methods:
                                continue
                            group_candidate_ids: Set[str] = set()
                            for method in active_group_methods:
                                group_candidate_ids |= query_method_candidates.get(method, set())
                            ablation_totals[group_name]["true"] += n_true
                            ablation_totals[group_name]["recovered"] += len(group_candidate_ids & true_set)

        ablation_results = []
        if config.collect_method_diagnostics:
            for group_name, totals in ablation_totals.items():
                active_group_methods = [m for m in ablation_groups[group_name] if m in config.methods]
                if not active_group_methods:
                    continue
                ablation_results.append({
                    "group": group_name,
                    "methods": active_group_methods,
                    "recall": (totals["recovered"] / totals["true"] if totals["true"] else None),
                })
            diagnostics["ablation_groups"] = ablation_results
            diagnostics["per_method_contribution"] = method_contribution
        else:
            diagnostics["method_diagnostics_skipped"] = True
            diagnostics["method_diagnostics_reason"] = "Disabled for memory-bounded validation."
            diagnostics["ablation_groups"] = []
            diagnostics["per_method_contribution"] = {}

    return result, diagnostics


def validate_country_agreement_streaming(
    gt_map: Dict[str, List[str]],
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    chunksize: int,
    batch_size: int,
) -> dict:
    s1_queries: List[QueryRecord] = []

    for batch in iter_source1_batches(
        s1_path,
        chunksize,
        batch_size,
        restrict_to=set(gt_map.keys()),
    ):
        s1_queries.extend(batch)

    s1_country_by_id = {
        query.entity_id: query.country_norm
        for query in s1_queries
    }

    del s1_queries
    gc.collect()

    wanted_s2: Set[str] = set()
    wanted_s3: Set[str] = set()

    for matched_ids in gt_map.values():
        for match_id in matched_ids:
            if match_id.startswith("S2-"):
                wanted_s2.add(match_id)
            elif match_id.startswith("S3-"):
                wanted_s3.add(match_id)

    target_country: Dict[str, Optional[str]] = {}

    for path, wanted_ids in (
        (s2_path, wanted_s2),
        (s3_path, wanted_s3),
    ):
        if not wanted_ids:
            continue

        remaining = set(wanted_ids)

        reader = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=True,
            chunksize=chunksize,
        )

        for chunk in reader:
            hit_mask = chunk["entity_id"].isin(remaining)

            if hit_mask.any():
                hits = chunk.loc[hit_mask]

                for row in hits.itertuples(index=False):
                    target_country[row.entity_id] = normalize_text(
                        getattr(row, "country", None)
                    )

                remaining -= set(hits["entity_id"].tolist())

            del chunk

            if not remaining:
                break

    same_country = 0
    different_country = 0
    missing_country = 0
    total = 0

    for s1_id, matched_ids in gt_map.items():
        s1_country = s1_country_by_id.get(s1_id)

        for match_id in matched_ids:
            total += 1
            target_value = target_country.get(match_id)

            if s1_country is None or target_value is None:
                missing_country += 1
            elif s1_country == target_value:
                same_country += 1
            else:
                different_country += 1

    return {
        "total_true_matches_checked": total,
        "same_country": same_country,
        "different_country": different_country,
        "missing_country_either_side": missing_country,
        "same_country_pct": (
            round(100.0 * same_country / total, 4)
            if total
            else None
        ),
        "different_country_pct": (
            round(100.0 * different_country / total, 4)
            if total
            else None
        ),
    }


class CandidatePairWriter:
    def __init__(
        self,
        wide_path: Path,
        long_path: Optional[Path] = None,
    ):
        self.wide_path = wide_path
        self.long_path = long_path

        self._wide_fh = open(
            wide_path,
            "w",
            encoding="utf-8",
            newline="",
        )

        self._wide_fh.write(
            "source1_entity_id\tcandidate_entity_ids\n"
        )

        self._long_fh = None

        if long_path is not None:
            self._long_fh = open(
                long_path,
                "w",
                encoding="utf-8",
                newline="",
            )

            self._long_fh.write(
                "source1_entity_id\tcandidate_entity_id\t"
                "source\tretrieval_methods\tretrieval_score\n"
            )

    def write(
        self,
        s1_entity_id: str,
        kept: List[
            Tuple[str, str, Set[str], Optional[float]]
        ],
    ) -> None:
        candidate_ids = [
            candidate_id
            for _source, candidate_id, _methods, _score in kept
        ]

        self._wide_fh.write(
            f"{s1_entity_id}\t{','.join(candidate_ids)}\n"
        )

        if self._long_fh is not None:
            for source, candidate_id, methods, score in kept:
                score_str = (
                    f"{score:.4f}"
                    if score is not None
                    else ""
                )

                methods_str = "|".join(sorted(methods))

                self._long_fh.write(
                    f"{s1_entity_id}\t{candidate_id}\t{source}\t"
                    f"{methods_str}\t{score_str}\n"
                )

    def close(self) -> None:
        self._wide_fh.close()

        if self._long_fh is not None:
            self._long_fh.close()


def write_report(
    report_path: Path,
    json_path: Path,
    sections: Dict[str, dict],
) -> None:
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("=" * 80 + "\n")
        report_file.write(
            "AMAZON ML CHALLENGE 2026 - CANDIDATE GENERATION REPORT\n"
        )
        report_file.write("=" * 80 + "\n\n")

        for title, content in sections.items():
            report_file.write("-" * 80 + "\n")
            report_file.write(f"{title}\n")
            report_file.write("-" * 80 + "\n")
            report_file.write(
                json.dumps(
                    content,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )
            report_file.write("\n\n")

    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(
            sections,
            json_file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


def recall_result_to_dict(result: RecallResult) -> dict:
    candidate_counts = sorted(result.candidate_counts)
    candidate_counts_before_cap = sorted(
        result.candidate_counts_before_cap
    )

    return {
        "n_queries": result.n_queries,
        "n_zero_match_queries": result.n_zero_match_queries,
        "n_with_truth": result.n_with_truth,
        "n_zero_candidate_queries": result.n_zero_candidate_queries,
        "pct_queries_with_zero_candidates": (
            round(
                100.0
                * result.n_zero_candidate_queries
                / result.n_queries,
                4,
            )
            if result.n_queries
            else None
        ),
        "overall_recall": result.overall_recall(),
        "recall_s2": result.recall_s2(),
        "recall_s3": result.recall_s3(),
        "true_matches_total": result.true_matches_total,
        "true_matches_recovered": result.true_matches_recovered,
        "candidate_count_distribution_after_cap": {
            "mean": (
                sum(candidate_counts) / len(candidate_counts)
                if candidate_counts
                else None
            ),
            "median": _percentile(candidate_counts, 50),
            "p90": _percentile(candidate_counts, 90),
            "p95": _percentile(candidate_counts, 95),
            "p99": _percentile(candidate_counts, 99),
            "max": max(candidate_counts) if candidate_counts else None,
        },
        "candidate_count_distribution_before_cap": {
            "mean": (
                sum(candidate_counts_before_cap)
                / len(candidate_counts_before_cap)
                if candidate_counts_before_cap
                else None
            ),
            "median": _percentile(candidate_counts_before_cap, 50),
            "p90": _percentile(candidate_counts_before_cap, 90),
            "p95": _percentile(candidate_counts_before_cap, 95),
            "p99": _percentile(candidate_counts_before_cap, 99),
            "max": (
                max(candidate_counts_before_cap)
                if candidate_counts_before_cap
                else None
            ),
        },
        "zero_match_entities": result.n_zero_match_queries,
        "zero_match_with_any_candidate": (
            result.zero_match_with_any_candidate
        ),
        "zero_match_candidate_presence_rate": (
            result.zero_match_with_any_candidate
            / result.n_zero_match_queries
            if result.n_zero_match_queries
            else None
        ),
        "recall_by_cardinality": {
            bucket: (
                result.per_cardinality_recovered[bucket]
                / result.per_cardinality_total[bucket]
                if result.per_cardinality_total[bucket]
                else None
            )
            for bucket in result.per_cardinality_total
        },
        "queries_by_cardinality": dict(
            result.per_cardinality_queries
        ),
        "difficult_case_recall": {
            tag: (
                result.difficult_case_recovered[tag]
                / result.difficult_case_total[tag]
                if result.difficult_case_total[tag]
                else None
            )
            for tag in result.difficult_case_total
        },
        "country_recall": {
            country: (
                result.country_recovered[country]
                / result.country_total[country]
                if result.country_total[country]
                else None
            )
            for country in result.country_total
        },
        "truncation_events": result.truncation_events,
        "truncation_lost_true_matches": (
            result.truncation_lost_true_matches
        ),
        "cap_possibly_affected_true_matches": (
            result.cap_possibly_affected_true_matches
        ),
    }


def run_demo(config: RetrievalConfig, output_dir: Path) -> bool:
    logger.info(
        "Running DEMO mode (synthetic in-memory data only; "
        "no real TSV files are read)"
    )

    s1_rows = [
        (
            "S1-001",
            "Orthopedic Associates Inc",
            "5101 Disbrow Court, Lincoln, NE",
            "US",
        ),
        (
            "S1-002",
            "KESTREL CHICAGO ASSOCIATES",
            "212 Sunkist Street, Ontario, CA",
            "US",
        ),
        (
            "S1-003",
            "White Producer Private Limited",
            "House No 418, Ghaziabad",
            "India",
        ),
        (
            "S1-004",
            "Keshav Exim",
            "D No 10, Chennai, Tamil Nadu",
            "India",
        ),
        (
            "S1-005",
            "Gertie's Colonial Holdings",
            None,
            "US",
        ),
        (
            "S1-006",
            "",
            "1 Empty Name St",
            "US",
        ),
        (
            "S1-007",
            "Unique Singleton Business Xyz123",
            "999 Nowhere Rd",
            "US",
        ),
        (
            "S1-008",
            "Common Word Company",
            "1 Somewhere Ave",
            "France",
        ),
        (
            "S1-009",
            "व्हाइट प्रोड्यूसर प्राइवेट लिमिटेड",
            None,
            "India",
        ),
    ]

    s2_rows = [
        (
            "S2-001",
            "Orthopedic Inc Associates",
            "DISBROW COURT, LINCON, NE",
            "US",
        ),
        (
            "S2-002",
            "Kestrel Chicago Asscgoiaims",
            "212 SUNKIST STREET, ONTARIO, CA",
            "US",
        ),
        (
            "S2-003",
            "keshav exim",
            "DOOR NO 10, Chennai, தமிழ்நாடு",
            "India",
        ),
        (
            "S2-004",
            "व्हाइट प्रोड्यूसर प्राइवेट लिमिटेड",
            "NO 163 HOUSE NO 418, GHAZIABAD",
            "India",
        ),
        (
            "S2-005",
            "Totally Unrelated Business",
            "123 Random Blvd",
            "US",
        ),
        (
            "S2-006",
            "Common Word Company Ltd",
            "1 Somewhere Avenue",
            "France",
        ),
    ]

    s3_rows = [
        (
            "S3-001",
            "Keshav Exim Corporation",
            "D No 10, Chennai, TN",
            "India",
        ),
        (
            "S3-002",
            "Common Word Trading Co",
            "2 Elsewhere St",
            "France",
        ),
    ]

    demo_dir = Path(
        tempfile.mkdtemp(prefix="candgen_demo_")
    )

    def write_tsv(
        path: Path,
        rows: List[Tuple[str, Optional[str], Optional[str], str]],
    ) -> None:
        with open(path, "w", encoding="utf-8") as file:
            file.write(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            )

            for entity_id, name, address, country in rows:
                file.write(
                    f"{entity_id}\t"
                    f"{name if name is not None else ''}\t"
                    f"{address if address is not None else ''}\t"
                    f"{country}\n"
                )

    demo_s1 = demo_dir / "s1.tsv"
    demo_s2 = demo_dir / "s2.tsv"
    demo_s3 = demo_dir / "s3.tsv"

    write_tsv(demo_s1, s1_rows)
    write_tsv(demo_s2, s2_rows)
    write_tsv(demo_s3, s3_rows)

    demo_ground_truth = {
        "S1-001": ["S2-001"],
        "S1-002": ["S2-002"],
        "S1-003": ["S2-004"],
        "S1-004": ["S2-003", "S3-001"],
        "S1-005": [],
        "S1-006": [],
        "S1-007": [],
        "S1-008": ["S2-006", "S3-002"],
        "S1-009": ["S2-004"],
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    demo_wide = output_dir / "_demo_candidate_pairs.tsv"
    demo_long = output_dir / "_demo_candidate_pairs_long.tsv"

    writer = CandidatePairWriter(
        demo_wide,
        demo_long,
    )

    try:
        build_diagnostics = run_build_pipeline(
            demo_s1,
            demo_s2,
            demo_s3,
            config,
            writer,
        )
    finally:
        writer.close()

    results_by_s1: Dict[str, List[str]] = {}

    with open(demo_wide, "r", encoding="utf-8") as file:
        next(file)

        for line in file:
            parts = line.rstrip("\n").split("\t")

            s1_id = parts[0]
            candidate_text = parts[1] if len(parts) > 1 else ""

            results_by_s1[s1_id] = [
                candidate_id
                for candidate_id in candidate_text.split(",")
                if candidate_id
            ]

    all_ok = True

    demo_report = {
        "per_entity": [],
        "assertions": [],
        "build_pipeline_diagnostics": build_diagnostics,
    }

    for s1_id, expected_matches in demo_ground_truth.items():
        candidates = results_by_s1.get(s1_id, [])

        found_expected = set(expected_matches).issubset(
            set(candidates)
        )

        demo_report["per_entity"].append(
            {
                "s1": s1_id,
                "candidates": candidates,
                "expected_true_matches": expected_matches,
                "all_expected_found": found_expected,
            }
        )

        if not found_expected:
            all_ok = False

            demo_report["assertions"].append(
                f"FAIL: {s1_id} expected {expected_matches} "
                f"but got {candidates}"
            )

    self_test_results = run_self_tests(config)

    demo_report["self_tests"] = self_test_results

    all_ok = all_ok and self_test_results["all_passed"]

    demo_report["all_assertions_passed"] = all_ok
    demo_report["transliteration_available"] = (
        TRANSLITERATION_AVAILABLE
    )
    demo_report["default_methods"] = DEFAULT_METHODS
    demo_report["all_methods"] = ALL_METHODS

    write_report(
        output_dir / "candidate_generation_report.txt",
        output_dir / "candidate_generation_report.json",
        {
            "DEMO_MODE_RESULTS": demo_report,
        },
    )

    for temporary_path in (demo_wide, demo_long):
        try:
            temporary_path.unlink()
        except OSError:
            pass

    try:
        for temporary_file in demo_dir.iterdir():
            temporary_file.unlink()

        demo_dir.rmdir()
    except OSError:
        pass

    logger.info(
        "DEMO complete. all_assertions_passed=%s "
        "(transliteration_available=%s)",
        all_ok,
        TRANSLITERATION_AVAILABLE,
    )

    for entry in demo_report["per_entity"]:
        logger.info(
            "  %s -> candidates=%s expected=%s ok=%s",
            entry["s1"],
            entry["candidates"],
            entry["expected_true_matches"],
            entry["all_expected_found"],
        )

    if not all_ok:
        logger.warning(
            "Some DEMO assertions or self-tests failed - "
            "see candidate_generation_report.json"
        )

    return all_ok


def _referenced_names_in_bytecode(func) -> Set[str]:
    names: Set[str] = set()

    try:
        instructions = list(dis.get_instructions(func))
    except TypeError:
        return names

    for instruction in instructions:
        if (
            instruction.opname
            in (
                "LOAD_GLOBAL",
                "LOAD_METHOD",
                "LOAD_NAME",
                "LOAD_DEREF",
                "LOAD_ATTR",
            )
            and instruction.argval
        ):
            names.add(instruction.argval)

    return names


def _walk_call_graph_for_forbidden_names(
    entry_func,
    forbidden_names: Set[str],
    module_globals: dict,
) -> Tuple[bool, Set[str], Set[str]]:
    visited: Set[str] = set()
    to_visit = [entry_func]
    forbidden_hits: Set[str] = set()

    while to_visit:
        current = to_visit.pop()
        function_name = getattr(current, "__name__", None)

        if function_name is None or function_name in visited:
            continue

        visited.add(function_name)

        referenced_names = _referenced_names_in_bytecode(current)

        for forbidden_name in forbidden_names:
            if forbidden_name in referenced_names:
                forbidden_hits.add(forbidden_name)

        for name in referenced_names:
            candidate = module_globals.get(name)

            if isinstance(candidate, types.FunctionType):
                if candidate.__name__ not in visited:
                    to_visit.append(candidate)

    return (
        len(forbidden_hits) > 0,
        visited,
        forbidden_hits,
    )


def test_build_test_has_no_ground_truth_dependency() -> bool:
    forbidden_found, visited, hits = (
        _walk_call_graph_for_forbidden_names(
            run_build_test,
            {
                "GROUND_TRUTH_FILE",
                "load_ground_truth_sample",
            },
            globals(),
        )
    )

    if forbidden_found:
        logger.warning(
            "build_test call graph (%s) unexpectedly references: %s",
            sorted(visited),
            sorted(hits),
        )

    return not forbidden_found


def run_self_tests(config: RetrievalConfig) -> dict:
    results: Dict[str, bool] = {}

    results["path_resolution_project_root_exists"] = (
        PROJECT_ROOT.exists()
    )

    try:
        for value in (
            None,
            float("nan"),
            "",
            "  ",
            42,
            ["x"],
        ):
            result = normalize_text(value)

            assert result is None or isinstance(result, str)

            if isinstance(result, str):
                assert result.strip().lower() not in (
                    "nan",
                    "none",
                    "null",
                )

        assert normalize_text(pd.NA) is None

        results["missing_value_handling"] = True
    except Exception:
        results["missing_value_handling"] = False

    try:
        devanagari = "व्हाइट प्रोड्यूसर प्राइवेट लिमिटेड"
        normalized = normalize_text(devanagari)

        assert normalized is not None
        assert "व" in normalized

        results["unicode_name_preserved"] = True
    except Exception:
        results["unicode_name_preserved"] = False

    try:
        tamil_address = "தமிழ்நாடு சென்னை"
        normalized_address = normalize_text(tamil_address)

        assert normalized_address is not None
        assert "த" in normalized_address

        results["unicode_address_preserved"] = True
    except Exception:
        results["unicode_address_preserved"] = False

    try:
        union = CandidateUnion()

        union.add("S2", "S2-1", "exact_name", 1.0)
        union.add("S2", "S2-1", "char_ngram", 0.7)

        items = list(union.items())

        assert len(items) == 1
        assert items[0][2] == {
            "exact_name",
            "char_ngram",
        }

        results["candidate_union_dedupe_and_provenance"] = True
    except Exception:
        results["candidate_union_dedupe_and_provenance"] = False

    try:
        store = RowStore()

        store.add_row("S2-A", "acme corp", "1 main st", "US")
        store.add_row("S2-B", "acme corp", "1 main st", "India")
        store.add_row("S2-C", "acme corp", "1 main st", None)

        config_country = RetrievalConfig(
            methods=["token_name"],
            token_df_ratio_threshold=1.0,
        )

        indexes = build_target_indexes(store, config_country)

        query_country_id = resolve_country_id(
            "us",
            store.country_to_id,
        )

        union = retrieve_candidates_for_entity(
            "acme corp",
            "acme corp",
            None,
            query_country_id,
            indexes,
            "S2",
            config_country,
        )

        found = union.entity_ids()

        assert "S2-A" in found
        assert "S2-B" not in found
        assert "S2-C" in found
        assert indexes.hit_counter.touched_rows() == []

        results["country_filter_before_increment_and_fallback"] = True
    except Exception:
        results["country_filter_before_increment_and_fallback"] = False

    try:
        items = [
            ("S2", f"S2-{i}", {"exact_name"}, 1.0)
            for i in range(100)
        ]

        kept, diagnostics = apply_final_safety_cap(
            items,
            cap=0,
        )

        assert len(kept) == 100
        assert diagnostics["truncated"] is False

        results["final_safety_cap_zero_means_no_cap"] = True
    except Exception:
        results["final_safety_cap_zero_means_no_cap"] = False

    try:
        items = [
            ("S2", f"S2-{i}", {"exact_name"}, 1.0)
            for i in range(100)
        ]

        items.append(
            (
                "S2",
                "S2-TRUEMATCH",
                {"char_ngram"},
                0.01,
            )
        )

        _kept, diagnostics = apply_final_safety_cap(
            items,
            cap=10,
            true_matched_set={"S2-TRUEMATCH"},
        )

        assert diagnostics["truncated"] is True
        assert diagnostics["true_matches_lost"] == 1

        results["truncation_loss_exact"] = True
    except Exception:
        results["truncation_loss_exact"] = False

    try:
        inverted = BoundedInvertedIndex(
            max_postings_per_key=2,
        )

        for i in range(5):
            inverted.add("common", i)

        inverted.add("rare", 99)

        diagnostics = inverted.diagnostics()

        assert diagnostics["capped_key_count"] == 1
        assert inverted.is_capped("common")
        assert not inverted.is_capped("rare")

        results["posting_cap_diagnostics"] = True
    except Exception:
        results["posting_cap_diagnostics"] = False

    try:
        rows = [
            (
                f"S1-{i}",
                f"S2-{i}" if i % 5 == 0 else "",
            )
            for i in range(500)
        ]

        ground_truth_dir = Path(tempfile.mkdtemp())
        ground_truth_path = ground_truth_dir / "gt_test.tsv"

        with open(ground_truth_path, "w", encoding="utf-8") as file:
            file.write(
                "source1_entity_id\tmatched_entity_ids\n"
            )

            for source1_id, matched_ids in rows:
                file.write(
                    f"{source1_id}\t{matched_ids}\n"
                )

        sample_a = load_ground_truth_sample(
            ground_truth_path,
            sample_size=20,
            seed=42,
        )

        sample_b = load_ground_truth_sample(
            ground_truth_path,
            sample_size=20,
            seed=42,
        )

        assert set(sample_a.keys()) == set(sample_b.keys())
        assert len(sample_a) == 20

        ground_truth_path.unlink()
        ground_truth_dir.rmdir()

        results["deterministic_reservoir_sampling"] = True
    except Exception:
        results["deterministic_reservoir_sampling"] = False

    try:
        results["build_test_has_no_ground_truth_dependency"] = (
            test_build_test_has_no_ground_truth_dependency()
        )
    except Exception:
        results["build_test_has_no_ground_truth_dependency"] = False

    try:
        batch_dir = Path(tempfile.mkdtemp())
        batch_path = batch_dir / "s1_batch_test.tsv"

        with open(batch_path, "w", encoding="utf-8") as file:
            file.write(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
            )

            for i in range(23):
                file.write(
                    f"S1-{i}\tName {i}\tAddr {i}\tUS\n"
                )

        batch_sizes = [
            len(batch)
            for batch in iter_source1_batches(
                batch_path,
                chunksize=10,
                batch_size=5,
            )
        ]

        assert all(size <= 5 for size in batch_sizes)
        assert sum(batch_sizes) == 23

        batch_path.unlink()
        batch_dir.rmdir()

        results["bounded_source1_batch_processing"] = True
    except Exception:
        results["bounded_source1_batch_processing"] = False

    try:
        store = RowStore()
        store.add_row("S2-X", "", "", "US")

        methods_for_test = (
            ALL_METHODS
            if TRANSLITERATION_AVAILABLE
            else [
                method
                for method in ALL_METHODS
                if not method.startswith("translit_")
            ]
        )

        indexes = build_target_indexes(
            store,
            RetrievalConfig(methods=methods_for_test),
        )

        retrieve_candidates_for_entity(
            "",
            "",
            "",
            -1,
            indexes,
            "S2",
            RetrievalConfig(
                methods=[
                    "exact_name",
                    "char_ngram",
                ]
            ),
        )

        results["empty_name_address_no_crash"] = True
    except Exception:
        results["empty_name_address_no_crash"] = False

    try:
        default_config = RetrievalConfig()

        assert "translit_char_ngram" not in DEFAULT_METHODS
        assert "translit_char_ngram" not in default_config.methods

        store = RowStore()
        store.add_row(
            "S2-TL1",
            "Some Business Name",
            "1 Main St",
            "US",
        )

        default_indexes = build_target_indexes(
            store,
            RetrievalConfig(methods=DEFAULT_METHODS),
        )

        assert default_indexes.translit_ngram_index is None
        assert "translit_ngram_index" not in default_indexes.built_structures

        explicit_indexes = build_target_indexes(
            store,
            RetrievalConfig(
                methods=[
                    "exact_name",
                    "translit_char_ngram",
                ]
            ),
        )

        assert "translit_ngram_index" in explicit_indexes.built_structures

        results["translit_default_exclusion_and_explicit_selection"] = True
    except Exception:
        results["translit_default_exclusion_and_explicit_selection"] = False

    try:
        counter = BoundedHitCounter(10)

        assert isinstance(counter.counts, array)
        assert not isinstance(counter.counts, dict)
        assert not isinstance(counter.counts, Counter)
        assert counter.counts.itemsize == 4

        results["hit_counter_array_backed_itemsize_4"] = True
    except Exception:
        results["hit_counter_array_backed_itemsize_4"] = False

    try:
        counter = BoundedHitCounter(10)

        counter.add(7)
        counter.add(7)
        counter.add(7)

        assert counter.get(7) == 3

        results["hit_counter_repeated_increment"] = True
    except Exception:
        results["hit_counter_repeated_increment"] = False

    try:
        counter = BoundedHitCounter(5)

        for row_id in (1, 2, 3):
            counter.add(row_id)

        for row_id in (2, 3, 4):
            counter.add(row_id)

        for row_id in (3, 4):
            counter.add(row_id)

        assert {
            row_id: counter.get(row_id)
            for row_id in (1, 2, 3, 4)
        } == {
            1: 1,
            2: 2,
            3: 3,
            4: 2,
        }

        results["hit_counter_multi_row_correctness"] = True
    except Exception:
        results["hit_counter_multi_row_correctness"] = False

    try:
        counter = BoundedHitCounter(10)

        counter.add(5)
        counter.add(5)
        counter.add(5)

        assert counter.touched_rows() == [5]

        counter.reset()

        assert counter.get(5) == 0
        assert counter.touched_rows() == []

        results["hit_counter_touched_uniqueness_and_reset"] = True
    except Exception:
        results["hit_counter_touched_uniqueness_and_reset"] = False

    try:
        counter = BoundedHitCounter(60_000)

        for row_id in range(60_000):
            counter.add(row_id)

        assert len(counter.touched_rows()) == 60_000
        assert all(
            counter.get(row_id) == 1
            for row_id in range(0, 60_000, 7919)
        )

        results["hit_counter_no_arbitrary_cap_60k"] = True
    except Exception:
        results["hit_counter_no_arbitrary_cap_60k"] = False

    try:
        counter = BoundedHitCounter(20)

        for row_id in [
            3,
            1,
            7,
            3,
            1,
            7,
            9,
            9,
            9,
            5,
        ]:
            counter.add(row_id)

        result_a = counter.top_k(3)

        counter.reset()

        for row_id in [
            3,
            1,
            7,
            3,
            1,
            7,
            9,
            9,
            9,
            5,
        ]:
            counter.add(row_id)

        result_b = counter.top_k(3)

        assert result_a == result_b
        assert result_a == [
            (9, 3),
            (1, 2),
            (3, 2),
        ]

        results["char_ngram_topk_deterministic_tiebreak"] = True
    except Exception:
        results["char_ngram_topk_deterministic_tiebreak"] = False

    try:
        store = RowStore()

        store.add_row(
            "S2-AD1",
            None,
            "5101 Disbrow Court, Lincoln, NE",
            "US",
        )

        address_config = RetrievalConfig(
            methods=[
                "token_address",
                "numeric_address",
                "char_ngram_address",
            ]
        )

        indexes = build_target_indexes(store, address_config)

        union = retrieve_candidates_for_entity(
            None,
            None,
            "5101 disbrow court lincoln ne",
            -1,
            indexes,
            "S2",
            address_config,
        )

        assert "S2-AD1" in union.entity_ids()

        results["address_methods_functional"] = True
    except Exception:
        results["address_methods_functional"] = False

    try:
        store2 = RowStore()
        store2.add_row(
            "S2-L1",
            "a",
            None,
            "US",
        )

        indexes2 = build_target_indexes(
            store2,
            RetrievalConfig(methods=["token_name"]),
        )

        assert indexes2.hit_counter is not None
        assert indexes2.hit_counter.size == len(store2) == 1

        del indexes2

        store3 = RowStore()

        for i in range(3):
            store3.add_row(
                f"S3-L{i}",
                "b",
                None,
                "US",
            )

        indexes3 = build_target_indexes(
            store3,
            RetrievalConfig(methods=["token_name"]),
        )

        assert indexes3.hit_counter is not None
        assert indexes3.hit_counter.size == len(store3) == 3

        results["s2_s3_reusable_counter_lifecycle"] = True
    except Exception:
        results["s2_s3_reusable_counter_lifecycle"] = False

    try:
        cfg = RetrievalConfig()
        assert cfg.collect_method_diagnostics is False
        results["retrieval_config_diagnostics_default_false"] = True
    except Exception:
        results["retrieval_config_diagnostics_default_false"] = False

    try:
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            s1_p = tdp / "s1.tsv"
            s2_p = tdp / "s2.tsv"
            s3_p = tdp / "s3.tsv"

            for p, rows in [
                (s1_p, [("S1-0", "A", "", "US")]),
                (s2_p, [("S2-0", "A", "", "US")]),
                (s3_p, [("S3-0", "A", "", "US")]),
            ]:
                with open(p, "w") as f:
                    f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                    for r in rows:
                        f.write("\t".join(r) + "\n")

            gt_map_mock = {"S1-0": ["S2-0", "S3-0"]}
            cfg = RetrievalConfig(methods=["exact_name"])
            res, diag = run_validation_pipeline(s1_p, s2_p, s3_p, gt_map_mock, cfg)
            assert res.n_queries == 1
        results["validation_pipeline_disk_backed_success"] = True
    except Exception:
        results["validation_pipeline_disk_backed_success"] = False

    results["all_passed"] = all(
        value
        for key, value in results.items()
        if key != "all_passed"
    )

    return results


def run_sample_validation(
    config: RetrievalConfig,
    output_dir: Path,
    full: bool = False,
) -> None:
    mode_label = "FULL VALIDATION" if full else "SAMPLE VALIDATION"

    logger.info(
        "Running %s | active methods (%d): %s | "
        "sample_size=%s | final_safety_cap=%s [0=NO CAP] | "
        "batch_size=%s",
        mode_label,
        len(config.methods),
        config.methods,
        "ALL (full validation)" if full else config.sample_size,
        config.final_safety_cap,
        config.batch_size,
    )

    if "translit_char_ngram" not in config.methods:
        logger.info(
            "NOTE: translit_char_ngram is NOT active "
            "(excluded by default for performance). Pass --methods all "
            "or include it explicitly in --methods to re-enable it."
        )

    logger.info(
        "Loading ground truth sample (two-pass, memory-bounded) ..."
    )

    gt_map = load_ground_truth_sample(
        GROUND_TRUTH_FILE,
        None if full else config.sample_size,
        config.seed,
    )

    logger.info(
        "Ground-truth QUERY entities in scope: %d "
        "(query-set size, NOT target-population size)",
        len(gt_map),
    )

    logger.info(
        "Validating country agreement from ground truth "
        "(streamed, sequential) ..."
    )

    country_stats = validate_country_agreement_streaming(
        gt_map,
        TRAIN_SOURCE_FILES["source1"],
        TRAIN_SOURCE_FILES["source2"],
        TRAIN_SOURCE_FILES["source3"],
        config.chunksize,
        config.batch_size,
    )

    logger.info("Country agreement: %s", country_stats)

    result, pipeline_diagnostics = run_validation_pipeline(
        TRAIN_SOURCE_FILES["source1"],
        TRAIN_SOURCE_FILES["source2"],
        TRAIN_SOURCE_FILES["source3"],
        gt_map,
        config,
    )

    logger.info(
        "Overall recall (all active methods): %s",
        result.overall_recall(),
    )

    if pipeline_diagnostics.get("ablation_groups"):
        for entry in pipeline_diagnostics["ablation_groups"]:
            logger.info(
                "  ablation[%s] methods=%s recall=%s",
                entry["group"],
                entry["methods"],
                entry["recall"],
            )

    sections = {
        "CONFIG": {
            **config.__dict__,
            "final_safety_cap_meaning": "0 = NO CAP",
            "translit_char_ngram_active": (
                "translit_char_ngram" in config.methods
            ),
        },
        "TRANSLITERATION_AVAILABLE": TRANSLITERATION_AVAILABLE,
        "DEFAULT_METHODS": DEFAULT_METHODS,
        "ALL_METHODS": ALL_METHODS,
        "QUERY_VS_TARGET_POPULATION_NOTE": {
            "query_entities_evaluated": len(gt_map),
            "note": (
                "This is the QUERY set size, not target population size. "
                "Target indexes are always built over the FULL required "
                "target population regardless of query sample size."
            ),
        },
        "COUNTRY_AGREEMENT_VALIDATION": country_stats,
        "PIPELINE_STAGE_DIAGNOSTICS": {
            key: value
            for key, value in pipeline_diagnostics.items()
            if key.startswith("stage_")
        },
        "POSTING_CAP_DIAGNOSTICS": (
            pipeline_diagnostics.get(
                "posting_cap_diagnostics",
                {},
            )
        ),
        "FULL_METHOD_RECALL": recall_result_to_dict(result),
    }

    if pipeline_diagnostics.get("method_diagnostics_skipped"):
        sections["METHOD_DIAGNOSTICS_SKIPPED"] = True
        sections["METHOD_DIAGNOSTICS_REASON"] = pipeline_diagnostics["method_diagnostics_reason"]

    sections["ABLATION_GROUPS"] = pipeline_diagnostics.get("ablation_groups", [])
    sections["PER_METHOD_CONTRIBUTION"] = pipeline_diagnostics.get("per_method_contribution", {})

    output_dir.mkdir(parents=True, exist_ok=True)

    write_report(
        output_dir / "candidate_generation_report.txt",
        output_dir / "candidate_generation_report.json",
        sections,
    )

    logger.info(
        "%s complete. Report written to %s",
        mode_label,
        output_dir / "candidate_generation_report.txt",
    )


def run_build_train(
    config: RetrievalConfig,
    output_dir: Path,
) -> None:
    logger.info(
        "Building TRAIN candidate pairs | active methods (%d): %s",
        len(config.methods),
        config.methods,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    writer = CandidatePairWriter(
        output_dir / "candidate_pairs_train.tsv",
        output_dir / "candidate_pairs_train_long.tsv",
    )

    try:
        summary = run_build_pipeline(
            TRAIN_SOURCE_FILES["source1"],
            TRAIN_SOURCE_FILES["source2"],
            TRAIN_SOURCE_FILES["source3"],
            config,
            writer,
        )
    finally:
        writer.close()

    logger.info("TRAIN candidates built: %s", summary)

    write_report(
        output_dir / "candidate_generation_report.txt",
        output_dir / "candidate_generation_report.json",
        {
            "BUILD_TRAIN_SUMMARY": summary,
        },
    )


def run_build_test(
    config: RetrievalConfig,
    output_dir: Path,
) -> None:
    """
    Generates TEST candidate pairs. Calls ONLY run_build_pipeline() against
    TEST_SOURCE_FILES - no code path to GROUND_TRUTH_FILE or
    load_ground_truth_sample exists here. Verified at runtime by
    test_build_test_has_no_ground_truth_dependency() via bytecode
    call-graph inspection.
    """
    logger.info(
        "Building TEST candidate pairs | active methods (%d): %s | "
        "ground truth file is never opened in this code path",
        len(config.methods),
        config.methods,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    writer = CandidatePairWriter(
        output_dir / "candidate_pairs_test.tsv",
        output_dir / "candidate_pairs_test_long.tsv",
    )

    try:
        summary = run_build_pipeline(
            TEST_SOURCE_FILES["source1"],
            TEST_SOURCE_FILES["source2"],
            TEST_SOURCE_FILES["source3"],
            config,
            writer,
        )
    finally:
        writer.close()

    logger.info("TEST candidates built: %s", summary)

    write_report(
        output_dir / "candidate_generation_report.txt",
        output_dir / "candidate_generation_report.json",
        {
            "BUILD_TEST_SUMMARY": summary,
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4: high-recall candidate generation / blocking."
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "demo",
            "sample_validation",
            "full_validation",
            "build_train",
            "build_test",
        ],
    )

    parser.add_argument(
        "--methods",
        type=str,
        default="default",
        help=(
            "Comma-separated method list, or 'default' "
            "(all methods except the expensive translit_char_ngram), "
            "or 'all' (every method including translit_char_ngram). "
            f"Valid methods: {','.join(ALL_METHODS)}"
        ),
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=15_000,
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=DEFAULT_CHUNKSIZE,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Bounded Source-1 batch size for build/validation pipelines.",
    )

    parser.add_argument(
        "--char-ngram-min",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--char-ngram-max",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--char-ngram-top-k",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--max-postings-per-gram",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--max-postings-per-token",
        type=int,
        default=4000,
    )

    parser.add_argument(
        "--final-safety-cap",
        type=int,
        default=0,
        help="0 = NO CAP (default everywhere; nonzero is explicitly opt-in).",
    )

    parser.add_argument(
        "--disable-country-partitioning",
        action="store_true",
    )

    parser.add_argument(
        "--collect-method-diagnostics",
        action="store_true",
        help="Enable memory-heavy per-method and ablation bookkeeping during validation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
    )

    return parser.parse_args()


def build_config_from_args(
    args: argparse.Namespace,
) -> RetrievalConfig:
    methods_arg = args.methods.strip().lower()

    if methods_arg == "all":
        methods = list(ALL_METHODS)
    elif methods_arg == "default":
        methods = list(DEFAULT_METHODS)
    else:
        methods = [
            method.strip()
            for method in args.methods.split(",")
            if method.strip()
        ]

        unknown_methods = set(methods) - set(ALL_METHODS)

        if unknown_methods:
            raise ValueError(
                f"Unknown retrieval method(s): {unknown_methods}. "
                f"Valid methods: {ALL_METHODS}"
            )

    if not TRANSLITERATION_AVAILABLE:
        skipped = [
            method
            for method in methods
            if method.startswith("translit_")
        ]

        if skipped:
            logger.warning(
                "unidecode not installed; transliteration methods "
                "unavailable and will be skipped: %s",
                skipped,
            )

        methods = [
            method
            for method in methods
            if not method.startswith("translit_")
        ]

    if (
        args.final_safety_cap > 0
        and args.mode in (
            "sample_validation",
            "full_validation",
        )
    ):
        logger.warning(
            "final_safety_cap=%d requested during %s; recall numbers "
            "WILL reflect truncation. Use 0 for uncapped recall measurement.",
            args.final_safety_cap,
            args.mode,
        )

    return RetrievalConfig(
        methods=methods,
        char_ngram_min=args.char_ngram_min,
        char_ngram_max=args.char_ngram_max,
        char_ngram_top_k=args.char_ngram_top_k,
        max_postings_per_gram=args.max_postings_per_gram,
        max_postings_per_token=args.max_postings_per_token,
        final_safety_cap=args.final_safety_cap,
        enable_country_partitioning=(
            not args.disable_country_partitioning
        ),
        chunksize=args.chunksize,
        batch_size=args.batch_size,
        sample_size=args.sample_size,
        seed=args.seed,
        collect_method_diagnostics=args.collect_method_diagnostics,
    )


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    output_dir = Path(args.output_dir)

    if args.mode == "demo":
        run_demo(config, output_dir)
    elif args.mode == "sample_validation":
        run_sample_validation(
            config,
            output_dir,
            full=False,
        )
    elif args.mode == "full_validation":
        run_sample_validation(
            config,
            output_dir,
            full=True,
        )
    elif args.mode == "build_train":
        run_build_train(config, output_dir)
    elif args.mode == "build_test":
        run_build_test(config, output_dir)
    else:
        raise ValueError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    main()