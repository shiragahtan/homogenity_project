"""Anything related to utility functions/CATE:
    - CATE
    - Expected CATE
"""

import functools
import logging
import time
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List
from xmlrpc.client import boolean
import attr
import pandas as pd
from z3 import *
import copy
import ast
from itertools import product
from itertools import chain, combinations
import random
from dowhy import CausalModel
import warnings
import pygraphviz as pgv

from prescription import Prescription

warnings.filterwarnings("ignore")
SRC_PATH = Path(__file__).parent.parent.parent
sys.path.append(os.path.join(SRC_PATH, "tools"))
from MutePrint import MutePrint

"""
This module contains utility functions for causal inference and treatment effect estimation.
It provides tools for generating and evaluating treatments, calculating conditional average
treatment effects (CATE), and solving optimization problems related to set coverage.
"""

THRESHOLD = 0.1
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s"
)


def CATE(df_g, DAG_str, treatments, attrOrdinal, tgtO):
    """
    Calculate the Conditional Average Treatment Effect (CATE) for a given treatment.

    Returns:
        float: The calculated CATE value, or 0 if the calculation fails or is insignificant.
    """
    ## ------------------- DAG Modification begins -------------------------
    # Add a new column named TempTreatment
    if len(df_g) == 0:
        return 0.0, 0.0
    df_g = df_g.copy()

    if attrOrdinal == None:
        keys = list(treatments.keys())
        vals = list(treatments.values())
        df_g["TempTreatment"] = (df_g[keys] == vals).all(axis=1)
    else:
        df_g["TempTreatment"] = df_g.apply(
            functools.partial(
                isTreatable, treatments=treatments, attrOrdinal=attrOrdinal
            ),
            axis=1,
        )
    DAG_str = DAG_after_treatments(DAG_str, treatments, tgtO)

    # remove graph name as dowhy doesn't support named graph string
    ## --------------------- DAG Modification ends -------------------------
    df_filtered = df_g[(df_g["TempTreatment"] == 0) | (df_g["TempTreatment"] == 1)]
    with MutePrint():
        model = CausalModel(
            data=df_filtered, graph=DAG_str, treatment="TempTreatment", outcome=tgtO
        )

        estimands = model.identify_effect()
        causal_estm_reg = model.estimate_effect(
            estimands,
            method_name="backdoor.linear_regression",
            target_units="ate",
            effect_modifiers=[],
            test_significance=True,
        )
    ATE = causal_estm_reg.value

    p_value = causal_estm_reg.test_stat_significance()["p_value"]
    if ATE == 0:
        logging.debug(f"Treatment: {treatments}, ATE: {ATE}")
        return -0.01, p_value
    if p_value > THRESHOLD:
        logging.debug(f"Treatment: {treatments}, ATE: {ATE}, p_value: {p_value}")
        return 0.0, p_value

    else:
        logging.debug(f"Treatment: {treatments}, ATE: {ATE}, p_value: {p_value}")
        return ATE, p_value


def isTreatable(record, treatments, attrOrdinal):
    """
    Checks record is treatable using the given treatments

    Returns:
        int: 1 if the row satisfies the treatment conditions, i.e. the
        treatment is effective, 0 otherwise.
    """
    # Each treatment {A:a1} = to check if A can be set to a1

    for treat_attr in treatments:
        if attrOrdinal == None and treat_attr in attrOrdinal:
            # In case ordinal_attr is defined
            # current value <p treatment value => treatment is not effective
            treat_rank = attrOrdinal[treat_attr][treatments[treat_attr]]
            record_rank = attrOrdinal[treat_attr][record[treat_attr]]
            if record_rank < treat_rank:
                return 0
        else:
            # In case ordinal_attr not defined
            # treatment value == current value => no effect on this tuple
            if record[treat_attr] != treatments[treat_attr]:
                return 0
    return 1


def DAG_after_treatments(DAG_str, treats: Dict, tgtO: str):
    """
    Modify the causal graph (DAG) to incorporate the treatment variable.

    """
    DAG = pgv.AGraph(DAG_str)

    # For all attributes treat,
    # replace edge `treat -> dep`  to `temp -> dep`
    # remove all edges `par->treat`
    nodes = set(DAG.nodes())
    treated_nodes = nodes.intersection(treats.keys())
    outEdges = DAG.out_edges(treats.keys())
    newOutEdges = set(map(lambda tup: ("TempTreatment", tup[1]), outEdges))
    # remove edges associated with treat nodes
    DAG.remove_nodes_from(treated_nodes)
    DAG.add_nodes_from(treated_nodes)

    DAG.add_edges_from(newOutEdges)  # add new nodes
    if not DAG.has_edge("TempTreatment", tgtO):
        DAG.add_edge("TempTreatment", tgtO)
    DAG_str = DAG.to_string()
    DAG_str = DAG_str.replace("\n", " ")
    return DAG_str
