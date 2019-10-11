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

from aaai20.exp import collect_results, collect_timings
from aaai20.io import (
    default_prefix_exp_fn_suffix,
    exp_directory,
    filename_dataset,
    filename_model,
    filename_query,
    filename_results,
    filename_timings,
)
from aaai20.utils import load_config
from mercs.utils.encoding import code_to_query, encode_attribute, get_att, query_to_code
from pxs.core.PxS import PxS

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

SCRIPT = Path(__file__).stem
RANDOM_STATE = 42

DEFAULT_FIT_CONFIG = dict()

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
    dn_tmp = exp_directory(exp_dname=exp_keyword, script=SCRIPT).get("tmp")
    if not os.path.exists(dn_tmp):
        os.makedirs(dn_tmp)

    clf = fit_pxs(dataset, cwd=dn_tmp, **fit_config)

    # Save model
    save_pxs(dataset, clf, keyword=model_keyword)

    return


# Actions
def fit_pxs(dataset, cwd=None, model_keyword="default", **fit_config):

    # Model fname
    suffix = ["pxs", model_keyword]
    fn_mod = filename_model(dataset, suffix=suffix, extension="xdsl")

    # Data fname
    suffix = ["train", "pxs"]
    fn_train = filename_dataset(dataset, step=2, suffix=suffix, extension="csv")

    clf = PxS()
    clf.fit(fn_train, cwd=cwd, model_fname=fn_mod, **fit_config)

    return clf


def save_pxs(dataset, classifier, keyword="default"):

    suffix = ["pxs", keyword]
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
