import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import dill as pkl
import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import f1_score

import mercs
from aaai20.exp import collect_results, collect_timings
from aaai20.io import (
    default_prefix_exp_fn_suffix,
    filename_dataset,
    filename_model,
    filename_query,
    filename_results,
    filename_timings,
)
from aaai20.utils import load_config
from mercs.core import Mercs
from mercs.utils.encoding import code_to_query, encode_attribute, get_att, query_to_code

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

SCRIPT = Path(__file__).stem
RANDOM_STATE = 42

DEFAULT_FIT_CONFIG = dict(
    random_state=RANDOM_STATE,
    classifier_algorithm="DT",
    regressor_algorithm="DT",
    clf_criterion="gini",
    rgr_criterion="mse",
    selection_algorithm="base",
    nb_targets=1,
    fraction_missing=0.0,
    nb_iterations=1,
    max_depth=8,
)

DEFAULT_CONFIG = dict(
    fit_config=dict(), model_keyword="default", exp_keyword="default", exp_idx=0
)


# Main function
def main(config_fname):

    # Load config + extract variables (in order of appearance)

    config = load_config(config_fname)
    config = {**DEFAULT_CONFIG, **config}

    dataset = config.get("dataset")
    model_keyword = config.get("model_keyword")
    exp_idx = config.get("exp_idx")
    exp_keyword = config.get("exp_keyword")

    fit_config = config.get("fit_config")
    fit_config = {**DEFAULT_FIT_CONFIG, **fit_config}

    # Make model
    clf = fit_mercs(dataset, **fit_config)

    msg = """
    dataset: {}
    nb_models: {}
    max_depth: {}
    """.format(dataset, len(clf.m_list), clf.m_list[0].max_depth)
    print(msg)

    # Save model
    save_mercs(dataset, clf, keyword=model_keyword)

    return


# Actions
def fit_mercs(dataset, **fit_config):

    # Load data
    fn_train = filename_dataset(dataset, step=2, suffix="train", extension="csv")
    df = pd.read_csv(fn_train, header=None, index_col=None)
    train = df.values
    train = train.astype(float)

    # Everything is nominal here
    nominal_ids = set(range(train.shape[1]))

    # Train
    msg = """
    Fit config = {}
    """.format(fit_config)
    print(msg)
    clf = Mercs(**fit_config)
    clf.fit(train, nominal_attributes=nominal_ids)

    return clf


def save_mercs(dataset, classifier, keyword="default"):

    suffix = ["mercs", keyword]
    fn_mod = filename_model(dataset, suffix=suffix)

    with open(fn_mod, "wb") as f:
        pkl.dump(classifier, f)
    return


# CLI
def create_parser():
    # Create the parser
    parser = argparse.ArgumentParser(
        description="Get the config filename and maybe the query index"
    )

    parser.add_argument(
        "--config_fname",
        action="store",
        type=str,
        required=True,
        help="Filename of configuration file.",
    )

    return parser


# For executable script
if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    config_fname_outer_scope = args.config_fname

    main(config_fname_outer_scope)
