"""
* **k independent random walks**
* **OPTIMIZED:** Uses pre-computed Numpy masks (Zero Setup Cost).
* **OPTIMIZED:** Bottom-Up Weighted Walk (No Apriori).
* **OPTIMIZED:** Lazy Evaluation (Checks delta on-the-fly).
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

# --- CONFIGURATION ---
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
NUM_WALKS = 1000

with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _CFG = json.load(fp)

BINARY_TREATMENT: str = _CFG["TREATMENT_COL"]
ATTRIBUTE_WEIGHTS_RAW: Dict[str, float] = _CFG.get("ATTRIBUTE_WEIGHTS", {})

# Normalize weights to ensure they work well as probabilities
if ATTRIBUTE_WEIGHTS_RAW:
    lo, hi = min(ATTRIBUTE_WEIGHTS_RAW.values()), max(ATTRIBUTE_WEIGHTS_RAW.values())
    div = hi - lo if hi != lo else 1.0
    ATTRIBUTE_WEIGHTS: Dict[str, float] = {
        a: (w - lo) / div for a, w in ATTRIBUTE_WEIGHTS_RAW.items()
    }
else:
    ATTRIBUTE_WEIGHTS = {}

sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe

class FastContext:
    """
    Optimized Backend: Holds pre-computed numpy masks.
    This replaces the slow Apriori setup. It converts the DataFrame into 
    a dictionary of boolean bitmaps for instant size checks.
    """
    def __init__(self, df: pd.DataFrame, treatment_col: str, outcome_col: str):
        self.n_rows = len(df)
        
        # 1. Extract Core Arrays for CATE Calculation
        self.treatment_arr = df[treatment_col].to_numpy()
        self.outcome_arr = df[outcome_col].to_numpy()
        
        # Calculate Global ATE (Baseline)
        try:
            self.full_ate = self._calc_cate_numpy(np.ones(self.n_rows, dtype=bool))
        except:
            self.full_ate = 0.0
        
        # 2. Pre-compute Attribute Masks
        # Structure: self.masks[ColumnName][Value] = NumpyBoolArray
        exclude = {treatment_col, BINARY_TREATMENT, outcome_col}
        self.attr_cols = [c for c in df.columns if c not in exclude]
        self.masks: Dict[str, Dict[Any, np.ndarray]] = {}
        
        for col in self.attr_cols:
            self.masks[col] = {}
            # Groupby is faster than iterating rows
            for val, indices in df.groupby(col).groups.items():
                mask = np.zeros(self.n_rows, dtype=bool)
                mask[indices] = True
                self.masks[col][val] = mask

    def _calc_cate_numpy(self, mask: np.ndarray) -> float:
        """
        Pure Numpy CATE calculation (Difference in Means).
        Runs in microseconds.
        """
        y = self.outcome_arr[mask]
        t = self.treatment_arr[mask]
        
        n1 = t.sum()
        n0 = len(t) - n1
        
        if n1 == 0 or n0 == 0:
            raise LinAlgError("Empty treatment group")
            
        mean_1 = y[t == 1].mean()
        mean_0 = y[t == 0].mean()
        return mean_1 - mean_0

def _weighted_random_walk_bottom_up(
    ctx: FastContext,
    delta: int,
    epsilon: float,
    k_walks: int = 500
) -> bool:
    """
    Lazy, Bottom-Up, Weighted Random Walk.
    
    1. Start Empty.
    2. Identify all neighbor attributes we can add where size > delta.
    3. Weight neighbors based on ATTRIBUTE_WEIGHTS (e.g. prefer Race/GINI).
    4. Randomly select one.
    5. Check CATE.
    """
    
    cate_count = 0
    
    # We run k independent walks
    for i in range(k_walks):
        
        # Reset state for new walk
        current_mask = np.ones(ctx.n_rows, dtype=bool)
        current_filter = {}
        used_cols = set()
        
        # WALK LOOP (Go deeper until dead end)
        while True:
            candidates = []
            candidate_weights = []
            candidate_vals = []
            candidate_masks = []
            
            # 1. Find valid neighbors (attributes we haven't used yet)
            available_cols = [c for c in ctx.attr_cols if c not in used_cols]
            
            # Optimization: Shuffle available cols to avoid index bias if weights are equal
            random.shuffle(available_cols)
            
            for col in available_cols:
                # Get the user-defined weight for this attribute
                # Default to small value if not in config
                w = ATTRIBUTE_WEIGHTS.get(col, 0.01)
                
                # Check all values for this column
                for val, val_mask in ctx.masks[col].items():
                    
                    # --- LAZY EVALUATION ---
                    # Check size INSTANTLY using bitwise AND
                    inter_mask = current_mask & val_mask
                    size = inter_mask.sum()
                    
                    # Only consider if size >= delta
                    if size >= delta:
                        candidates.append(col)
                        candidate_vals.append(val)
                        candidate_masks.append(inter_mask)
                        
                        # Add small jitter so weights aren't perfectly identical
                        candidate_weights.append(w + random.uniform(0, 0.05))
            
            # Dead End check
            if not candidates:
                break
            
            # 2. WEIGHTED RANDOM SELECTION
            # This makes it a Random Walk (Probabilistic), not Greedy (Deterministic)
            total_w = sum(candidate_weights)
            probs = [x / total_w for x in candidate_weights]
            
            # Pick ONE based on probability
            choice_idx = random.choices(range(len(candidates)), weights=probs, k=1)[0]
            
            # 3. Move to that node
            col_choice = candidates[choice_idx]
            val_choice = candidate_vals[choice_idx]
            
            current_mask = candidate_masks[choice_idx]
            current_filter[col_choice] = val_choice
            used_cols.add(col_choice)
            
            # 4. Check CATE
            try:
                cate = ctx._calc_cate_numpy(current_mask)
                cate_count += 1
                
                if abs(cate - ctx.full_ate) > epsilon:
                    # --- Violation Found ---
                    print(f"Stopping early: Calculated CATE {cate_count} times.")
                    print(f"Breaking Subgroup: {current_filter}")
                    return False 
            except LinAlgError:
                pass

            # 5. Random Restart (Stochastic Jump)
            # 10% chance to stop this specific walk and start a new one from top
            # This prevents getting stuck in deep, non-interesting branches.
            if len(used_cols) > 3 and random.random() < 0.10:
                break

    print(f"RW Finished. Checked {cate_count} subgroups across {k_walks} walks. No violation.")
    return True

# Public API
def calc_utility_for_subgroups(
    mode: int,
    algorithm: Callable, 
    df: pd.DataFrame,
    treatment_col: str,
    delta: int,
    epsilon: float,
    *,
    outcome_col: Optional[str] = None,
    tgtO: Optional[str] = None,
    k_walks: int = NUM_WALKS,
    **kwargs: object,
):
    """
    Drop-in replacement for existing API.
    Ignores 'algorithm' (Apriori) because we use optimized internal logic.
    """
    outcome_col = outcome_col or tgtO
    if outcome_col is None:
        raise ValueError("Need outcome_col / tgtO")

    if mode == 0:
        # 1. Initialize Fast Backend (One-time cost, ~0.05s)
        ctx = FastContext(df, treatment_col, outcome_col)
        
        # 2. Run True Weighted Random Walk
        return _weighted_random_walk_bottom_up(ctx, delta, epsilon, k_walks=k_walks)

    # Fallback for "All Subgroups" mode (Mode 1)
    # Since RW is designed for finding ONE violation, Mode 1 is not its strength.
    # We return empty to prevent crashes if selected by accident.
    print("Warning: Optimized RW only implements Homogeneity Check (Mode 0).")
    return [], 0