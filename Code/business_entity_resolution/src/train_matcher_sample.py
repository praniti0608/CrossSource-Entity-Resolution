from __future__ import annotations

import os
import sys
import gc
import time
import random
import numpy as np
import pandas as pd
from pathlib import Path
import lightgbm as lgb
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Tuple, List

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
FEATURES_FILE = PROJECT_ROOT / "output" / "features_sample.tsv"
GROUND_TRUTH_FILE = PROJECT_ROOT / "dataset" / "train" / "train_ground_truth.tsv"

TRAIN_SOURCE_FILES = {
    "source1": PROJECT_ROOT / "dataset" / "train" / "train_source1.tsv",
    "source2": PROJECT_ROOT / "dataset" / "train" / "train_source2.tsv",
    "source3": PROJECT_ROOT / "dataset" / "train" / "train_source3.tsv",
}

EXPECTED_FEATURES = [
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
    "cg_address_token"
]

def get_1000_s1_sample(seed: int = 42) -> List[str]:
    rng = random.Random(seed)
    reservoir = []
    n_seen = 0
    
    reader = pd.read_csv(TRAIN_SOURCE_FILES["source1"], sep="\t", usecols=["entity_id"], chunksize=200000, dtype=str)
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

def calculate_macro_f05(val_df: pd.DataFrame, val_s1_ids: set, gt_map: dict, threshold: float, prefix: str = None) -> float:
    scores = []
    
    pred_mask = val_df["pred_prob"] >= threshold
    if prefix:
        pred_mask = pred_mask & val_df["candidate_entity_id"].str.startswith(prefix)
        
    pred_groups = {}
    for s1_id, group in val_df[pred_mask].groupby("source1_entity_id", observed=True):
        pred_groups[str(s1_id)] = set(group["candidate_entity_id"])
        
    for s1_id in val_s1_ids:
        true_set = gt_map.get(s1_id, set())
        if prefix:
            true_set = {x for x in true_set if x.startswith(prefix)}
            
        pred_set = pred_groups.get(s1_id, set())
        
        if not true_set and not pred_set:
            scores.append(1.0)
        elif not true_set and pred_set:
            scores.append(0.0)
        elif true_set and not pred_set:
            scores.append(0.0)
        else:
            intersection = true_set.intersection(pred_set)
            if not intersection:
                scores.append(0.0)
            else:
                precision = len(intersection) / len(pred_set)
                recall = len(intersection) / len(true_set)
                f05 = (1.25 * precision * recall) / ((0.25 * precision) + recall)
                scores.append(f05)
                
    if not scores:
        return 0.0
    return float(np.mean(scores))

def calculate_source_metrics(src_df: pd.DataFrame, threshold: float) -> Tuple[float, float, float, int, int]:
    actual_pos = (src_df["label"] == 1)
    pred_pos = (src_df["pred_prob"] >= threshold)
    
    true_pos_count = (actual_pos & pred_pos).sum()
    pred_pos_count = pred_pos.sum()
    actual_pos_count = actual_pos.sum()
    
    precision = true_pos_count / pred_pos_count if pred_pos_count > 0 else 0.0
    recall = true_pos_count / actual_pos_count if actual_pos_count > 0 else 0.0
    
    if precision > 0 or recall > 0:
        f05 = (1.25 * precision * recall) / ((0.25 * precision) + recall)
    else:
        f05 = 0.0
        
    return precision, recall, f05, actual_pos_count, pred_pos_count

def run_experiment():
    print("============================================================")
    print("PHASE 3: LIGHTGBM TRAINING EXPERIMENT")
    print("============================================================")
    
    if not FEATURES_FILE.exists():
        print(f"Error: Features file not found at {FEATURES_FILE}")
        sys.exit(1)
        
    if not GROUND_TRUTH_FILE.exists():
        print(f"Error: Ground truth file not found at {GROUND_TRUTH_FILE}")
        sys.exit(1)
        
    print("\n0. RECONSTRUCTING 1000-S1 SAMPLE & FULL GROUND TRUTH...")
    sampled_s1_ids_list = get_1000_s1_sample(seed=42)
    sampled_s1_ids = set(sampled_s1_ids_list)
    
    gt_map = {eid: set() for eid in sampled_s1_ids}
    gt_reader = pd.read_csv(GROUND_TRUTH_FILE, sep="\t", chunksize=200000, dtype=str, keep_default_na=False)
    for chunk in gt_reader:
        mask = chunk["source1_entity_id"].isin(sampled_s1_ids)
        if mask.any():
            for _, row in chunk[mask].iterrows():
                s1_id = str(row["source1_entity_id"]).strip()
                raw_matched = str(row.get("matched_entity_ids", "")).strip()
                if raw_matched:
                    gt_map[s1_id].update({m.strip() for m in raw_matched.split(",") if m.strip()})
                    
    print("\n1. LOADING AND VALIDATING DATA...")
    
    dtypes = {
        "source1_entity_id": "category",
        "candidate_entity_id": "string",
        "label": "int8",
        "name_exact_match": "int8",
        "name_char_similarity": "float32",
        "name_jaccard": "float32",
        "name_containment": "float32",
        "name_shared_token_count": "int16",
        "name_token_count_diff": "int16",
        "name_len_diff": "int16",
        "addr_exact_match": "int8",
        "addr_char_similarity": "float32",
        "addr_jaccard": "float32",
        "addr_containment": "float32",
        "addr_shared_token_count": "int16",
        "addr_token_count_diff": "int16",
        "addr_len_diff": "int16",
        "addr_shared_numeric_count": "int16",
        "country_exact_match": "int8",
        "cg_exact_name": "int8",
        "cg_exact_address": "int8",
        "cg_name_token": "int8",
        "cg_address_token": "int8",
    }
    
    df = pd.read_csv(FEATURES_FILE, sep="\t", dtype=dtypes)
    
    represented_s1_count = df["source1_entity_id"].nunique()
    missing_s1_count = len(sampled_s1_ids_list) - represented_s1_count
    
    missing_cols = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing feature columns: {missing_cols}")
        sys.exit(1)
        
    if not df["label"].isin([0, 1]).all():
        print("Error: Label column contains values other than 0 and 1.")
        sys.exit(1)
        
    null_counts = df[EXPECTED_FEATURES].isnull().sum()
    if null_counts.sum() > 0:
        print("Error: NaN values found in features.")
        sys.exit(1)
        
    if np.isinf(df[EXPECTED_FEATURES].select_dtypes(include=[np.number])).values.any():
        print("Error: Infinite values found in features.")
        sys.exit(1)
        
    total_pos = (df["label"] == 1).sum()
    total_neg = (df["label"] == 0).sum()
    print(f"  Total rows: {len(df):,}")
    print(f"  Total positives: {total_pos:,}")
    print(f"  Total negatives: {total_neg:,}")
    
    if total_pos == 0 or total_neg == 0:
        print("Error: Both classes must be present.")
        sys.exit(1)
        
    print("\n2. GROUPED TRAIN/VALIDATION SPLIT...")
    unique_s1 = np.array(sampled_s1_ids_list)
    np.random.seed(42)
    shuffled_s1 = np.random.permutation(unique_s1)
    
    split_idx = int(len(shuffled_s1) * 0.8)
    train_s1_ids = set(shuffled_s1[:split_idx])
    val_s1_ids = set(shuffled_s1[split_idx:])
    
    intersection = train_s1_ids.intersection(val_s1_ids)
    if intersection:
        print("Error: Data leakage detected between train and val splits.")
        sys.exit(1)
        
    train_mask = df["source1_entity_id"].isin(train_s1_ids)
    train_df = df[train_mask].copy()
    val_df = df[~train_mask].copy()
    
    del df
    gc.collect()
    
    train_pos = (train_df["label"] == 1).sum()
    train_neg = (train_df["label"] == 0).sum()
    val_pos = (val_df["label"] == 1).sum()
    val_neg = (val_df["label"] == 0).sum()

    if train_pos == 0 or train_neg == 0:
        print("Error: Training split does not contain both classes.")
        sys.exit(1)

    if val_pos == 0 or val_neg == 0:
        print("Error: Validation split does not contain both classes.")
        sys.exit(1)
    
    print(f"  Train rows: {len(train_df):,}")
    print(f"  Validation rows: {len(val_df):,}")
    print(f"  Train Source-1 entities: {len(train_s1_ids):,}")
    print(f"  Validation Source-1 entities: {len(val_s1_ids):,}")
    print(f"  Train positives: {train_pos:,}")
    print(f"  Train negatives: {train_neg:,}")
    print(f"  Validation positives: {val_pos:,}")
    print(f"  Validation negatives: {val_neg:,}")
    
    print("\n3. MEMORY SAFETY ENSURED (Df copied efficiently and original deleted).")
    
    print("\n4. CLASS IMBALANCE...")
    # scale_pos_weight is removed for this experiment
    print(f"  scale_pos_weight removed for this run.")
    
    print("\n5. LIGHTGBM MODEL TRAINING...")
    X_train = train_df[EXPECTED_FEATURES]
    y_train = train_df["label"]
    
    X_val = val_df[EXPECTED_FEATURES]
    y_val = val_df["label"]
    
    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.05,
        n_estimators=500,
        num_leaves=31,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    
    fit_kwargs = {"eval_set": [(X_val, y_val)], "eval_metric": "auc"}
    try:
        from lightgbm.callback import early_stopping
        fit_kwargs["callbacks"] = [early_stopping(stopping_rounds=50, verbose=False)]
    except ImportError:
        fit_kwargs["early_stopping_rounds"] = 50
        
    t0 = time.time()
    model.fit(X_train, y_train, **fit_kwargs)
    print(f"  Training completed in {time.time() - t0:.2f} seconds.")

    model_path = PROJECT_ROOT / "output" / "lgbm_model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")
    
    print("\n6. VALIDATION PREDICTIONS...")
    val_df["pred_prob"] = model.predict_proba(X_val)[:, 1]
    
    min_prob = val_df["pred_prob"].min()
    max_prob = val_df["pred_prob"].max()
    if min_prob < 0.0 or max_prob > 1.0:
        print("Error: Predicted probabilities outside [0, 1].")
        sys.exit(1)
        
    print("\n7. PREDICTION PROBABILITY DIAGNOSTICS...")
    pred_probs = val_df["pred_prob"]
    print(f"  Min: {pred_probs.min():.6f}")
    print(f"  Max: {pred_probs.max():.6f}")
    print(f"  Mean: {pred_probs.mean():.6f}")
    print(f"  Median: {pred_probs.median():.6f}")
    print(f"  90th percentile: {np.percentile(pred_probs, 90):.6f}")
    print(f"  95th percentile: {np.percentile(pred_probs, 95):.6f}")
    print(f"  99th percentile: {np.percentile(pred_probs, 99):.6f}")
    print(f"  99.5th percentile: {np.percentile(pred_probs, 99.5):.6f}")
    print(f"  99.9th percentile: {np.percentile(pred_probs, 99.9):.6f}")
    print(f"  Unique probability values: {pred_probs.nunique():,}")
    
    print("\n  Validation rows satisfying:")
    print(f"    prob >= 0.01: {(pred_probs >= 0.01).sum():,}")
    print(f"    prob >= 0.05: {(pred_probs >= 0.05).sum():,}")
    print(f"    prob >= 0.10: {(pred_probs >= 0.10).sum():,}")
    print(f"    prob >= 0.25: {(pred_probs >= 0.25).sum():,}")
    print(f"    prob >= 0.50: {(pred_probs >= 0.50).sum():,}")
    print(f"    prob >= 0.75: {(pred_probs >= 0.75).sum():,}")
    print(f"    prob >= 0.90: {(pred_probs >= 0.90).sum():,}")
    print(f"    prob >= 0.95: {(pred_probs >= 0.95).sum():,}")
    print(f"    prob >= 0.99: {(pred_probs >= 0.99).sum():,}")
    
    print("\n  [Candidate Generation Ceiling Diagnostics]")
    total_true_matches = 0
    recovered_true_matches = 0
    
    val_candidate_groups = {}
    for s1_id, group in val_df.groupby("source1_entity_id", observed=True):
        val_candidate_groups[str(s1_id)] = set(group["candidate_entity_id"])
        
    for s1_id in val_s1_ids:
        true_set = gt_map.get(s1_id, set())
        cand_set = val_candidate_groups.get(s1_id, set())
        
        total_true_matches += len(true_set)
        recovered_true_matches += len(true_set.intersection(cand_set))
        
    candidate_recall = recovered_true_matches / total_true_matches if total_true_matches > 0 else 0.0
    
    ceiling_f05 = calculate_macro_f05(
        val_df.assign(pred_prob=1.0),
        val_s1_ids,
        gt_map,
        0.5
    )
    
    print(f"  Candidate-generation ceiling (all candidates predicted positive): {ceiling_f05:.4f}")
    print(f"  Validation true matches: {total_true_matches}")
    print(f"  Validation recovered by candidates: {recovered_true_matches}")
    print(f"  Validation candidate recall: {candidate_recall:.4f}")

    print("\n8. THRESHOLD SEARCH (Using FULL Ground Truth)...")
    print("  [Coarse Search]")
    best_coarse_th = 0.5
    best_coarse_f05 = -1.0
    
    for th in np.arange(0.05, 1.00, 0.05):
        f05 = calculate_macro_f05(val_df, val_s1_ids, gt_map, th)
        if f05 >= best_coarse_f05:
            best_coarse_f05 = f05
            best_coarse_th = th
            
    print(f"  Best coarse threshold: {best_coarse_th:.2f} (F0.5: {best_coarse_f05:.4f})")
    
    print("  [Fine Search]")
    start_th = max(0.01, best_coarse_th - 0.05)
    end_th = min(0.99, best_coarse_th + 0.05)
    
    best_th = best_coarse_th
    best_macro_f05 = -1.0
    
    print(f"  threshold | macro_F0.5 | overall_precision | overall_recall | predicted_positive_pairs")
    print(f"  ----------------------------------------------------------------------------------------")
    
    for th in np.arange(start_th, end_th + 0.001, 0.01):
        th = round(th, 2)
        f05 = calculate_macro_f05(val_df, val_s1_ids, gt_map, th)
        
        pred_pos_count = (val_df["pred_prob"] >= th).sum()
        true_pos_overall = ((val_df["label"] == 1) & (val_df["pred_prob"] >= th)).sum()
        actual_pos_overall = (val_df["label"] == 1).sum()
        
        overall_prec = true_pos_overall / pred_pos_count if pred_pos_count > 0 else 0.0
        overall_rec = true_pos_overall / actual_pos_overall if actual_pos_overall > 0 else 0.0
        
        print(f"  {th:9.2f} | {f05:10.4f} | {overall_prec:17.4f} | {overall_rec:14.4f} | {pred_pos_count}")
        
        if f05 >= best_macro_f05:
            best_macro_f05 = f05
            best_th = th
            
    best_pred_pos_count = (val_df["pred_prob"] >= best_th).sum()
    best_true_pos_overall = ((val_df["label"] == 1) & (val_df["pred_prob"] >= best_th)).sum()
    best_actual_pos_overall = (val_df["label"] == 1).sum()
    best_overall_precision = best_true_pos_overall / best_pred_pos_count if best_pred_pos_count > 0 else 0.0
    best_overall_recall = best_true_pos_overall / best_actual_pos_overall if best_actual_pos_overall > 0 else 0.0
    
    print("\n9. ADDITIONAL VALIDATION METRICS...")
    roc_auc = roc_auc_score(val_df["label"], val_df["pred_prob"])
    pr_auc = average_precision_score(val_df["label"], val_df["pred_prob"])
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Average Precision (PR-AUC): {pr_auc:.4f}")
    
    print("\n10. SOURCE-SPECIFIC ANALYSIS...")
    s2_mask = val_df["candidate_entity_id"].str.startswith("S2-")
    s3_mask = val_df["candidate_entity_id"].str.startswith("S3-")
    
    s2_df = val_df[s2_mask]
    s3_df = val_df[s3_mask]
    
    s2_macro_f05 = calculate_macro_f05(val_df, val_s1_ids, gt_map, best_th, prefix="S2-")
    print("  [S2 Candidates]")
    print(f"    S2 entity-level Macro F0.5: {s2_macro_f05:.4f}")
    
    s3_macro_f05 = calculate_macro_f05(val_df, val_s1_ids, gt_map, best_th, prefix="S3-")
    print("  [S3 Candidates]")
    print(f"    S3 entity-level Macro F0.5: {s3_macro_f05:.4f}")
    
    print("\n11. FEATURE IMPORTANCE...")
    importance = model.booster_.feature_importance(importance_type="gain")
    feature_names = X_train.columns
    feat_imp = pd.DataFrame({"Feature": feature_names, "Gain": importance})
    feat_imp = feat_imp.sort_values(by="Gain", ascending=False).head(15)
    
    for _, row in feat_imp.iterrows():
        print(f"  {row['Feature']:30s} : {row['Gain']:.4f}")
        
    print("\n12. SANITY CHECKS...")
    unique_cand_map = val_df.groupby("candidate_entity_id")["source1_entity_id"].nunique()
    if (unique_cand_map > 1).any():
        print("Warning: Some candidates map to multiple S1 entities in validation (expected if data contains many-to-many, but metric handles this per S1).")
        
    print(f"  Total sampled S1 entities: {len(sampled_s1_ids_list)}")
    print(f"  S1 entities represented in features_sample.tsv: {represented_s1_count}")
    print(f"  S1 entities with zero candidate rows: {missing_s1_count}")
    print(f"  Validation S1 entities evaluated by Macro F0.5: {len(val_s1_ids)}")
    
    print("\n============================================================")
    print("PHASE 3 SAMPLE TRAINING COMPLETE")
    print("============================================================")
    print(f"Best validation threshold: {best_th:.2f}")
    print(f"Best validation Macro F0.5: {best_macro_f05:.4f}")
    print(f"Overall pair precision: {best_overall_precision:.4f}")
    print(f"Overall pair recall: {best_overall_recall:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"Average Precision: {pr_auc:.4f}")
    print("\nThis is a development validation result on the 1000-S1 sample.")
    print("and must NOT be interpreted as leaderboard performance.")

if __name__ == "__main__":
    run_experiment()