from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Set, Any

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

# --- CONFIGURATION ---
# Load config to get Treatment Column
CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "config.json"
with open(CONFIG_PATH, "r", encoding="utf-8") as fp:
    _CFG = json.load(fp)

BINARY_TREATMENT: str = _CFG["TREATMENT_COL"]

# Import ATE calculation helper
sys.path.append(str(Path(__file__).resolve().parent.parent / "yarden_files"))
from ATE_update import calculate_ate_safe


def calc_utility_for_subgroups(
        mode: int,
        df: pd.DataFrame,
        treatment_col: str,
        delta: int,
        epsilon: float,
        utility_all: float,
        *,
        tgtO: Optional[str] = None,
        **kwargs: object,
):
    if mode == 0:
        pass
        # TODO: implement causal forest

    return [], 0
