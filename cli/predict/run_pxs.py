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
    exp_directory,
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

DEFAULT_PREDICT_CONFIG = dict(inference_algorithm="EPIS_sampling")

DEFAULT_CONFIG = dict(
    predict_config=dict(),
    model_keyword="default",
    exp_keyword="default",
    qry_idx=None,
    exp_idx=0,
)


# Main function
def main(config_fname, q_idx):

    # Load config + extract variables (in order of appearance)
    qry_idx = q_idx

    config = load_config(config_fname)
    config = {**DEFAULT_CONFIG, **config}

    dataset = config.get("dataset")
    model_keyword = config.get("model_keyword")
    exp_idx = config.get("exp_idx")
    exp_keyword = config.get("exp_keyword")

    predict_config = config.get("predict_config")
    predict_config = {**DEFAULT_PREDICT_CONFIG, **predict_config}

    # Load model
    clf = load_pxs(dataset, keyword=model_keyword)

    cwd = exp_directory(exp_dname=exp_keyword, script=SCRIPT).get("tmp")
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    clf.set_cwd(cwd)

    # Predictions
    q_idx_return, q_codes_return, results, timings = predict_pxs(
        dataset, clf, q_idx=qry_idx, **predict_config
    )

    # Post-process outputs
    df_results = collect_results(
        dataset,
        q_codes_return,
        results,
        q_idx=q_idx_return,
        identifier=exp_keyword,
        exp_idx=exp_idx,
    )

    df_timings = collect_timings(timings, q_idx=q_idx_return)

    # Save results
    prefix, fn_exp, suffix = default_prefix_exp_fn_suffix(
        config,
        predict_config=predict_config,
        exp_idx=exp_idx,
        qry_idx=qry_idx,
        exp_fn_fields=["inference_algorithm"],
    )

    fn_res = filename_results(
        exp_fname=fn_exp,
        exp_dname=exp_keyword,
        prefix=prefix,
        suffix=suffix,
        script=SCRIPT,
    )
    df_results.to_csv(fn_res, index=False)

    # Save timings
    fn_tmg = filename_timings(
        exp_fname=fn_exp,
        exp_dname=exp_keyword,
        prefix=prefix,
        suffix=suffix,
        script=SCRIPT,
    )
    df_timings.to_csv(fn_tmg, index=False)

    return


# Actions
def predict_pxs(dataset, classifier, q_idx=None, **predict_config):
    # Init
    f1_micro = []
    f1_macro = []
    q_codes_return = []
    q_idx_return = []
    inf_time = []

    default_predict_config = dict(inference_algorithm="EPIS_sampling")
    predict_config = {**default_predict_config, **predict_config}

    # Load queries
    fn_qry = filename_query(dataset, suffix="default")
    q_codes = np.load(fn_qry)

    # Load data
    suffix = "test-pxs"
    fn_test = filename_dataset(dataset, step=2, suffix=suffix, extension="csv")
    df = pd.read_csv(fn_test, index_col=None)
    test = df.values
    test = test.astype(float)

    # Load ind_time
    ind_time = classifier.s["model_data"].get("ind_time")

    # Filter for query_idx
    include = {
        type(None): lambda x: True,
        list: lambda x: x in q_idx,
        int: lambda x: x == q_idx,
    }
    ok = include[type(q_idx)]

    for query_idx, q_code in enumerate(q_codes):
        if ok(query_idx):
            q_idx_return.append(query_idx)
            q_codes_return.append(q_code)

            q_f1_micro, q_f1_macro, q_inf_time = single_query_predict(
                classifier, test, fn_test, query_idx, q_code, **predict_config
            )

            inf_time.append(q_inf_time)
            f1_micro.append(q_f1_micro)
            f1_macro.append(q_f1_macro)

    q_codes_return = np.vstack(q_codes_return)
    results = dict(f1_micro=f1_micro, f1_macro=f1_macro)
    timings = dict(ind_time=ind_time, inf_time=inf_time)

    return q_idx_return, q_codes_return, results, timings


def single_query_predict(classifier, test, fn_test, q_idx, q_code, **predict_config):
    query_config = dict(
        miss_idx=get_att(q_code, kind="miss").tolist(),
        targ_idx=get_att(q_code, kind="targ").tolist(),
        q_idx=q_idx,
    )

    y_true = test[:, query_config["targ_idx"]]
    y_pred = classifier.predict(fn_test, **query_config, **predict_config)

    q_inf_time = classifier.s["model_data"].get("inf_time")

    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return f1_micro, f1_macro, q_inf_time


def load_pxs(dataset, keyword="default"):
    suffix = ["pxs", keyword]
    fn_mod = filename_model(dataset, suffix=suffix)

    with open(fn_mod, "rb") as f:
        clf = pkl.load(f)
    return clf


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

    parser.add_argument(
        "--q_idx",
        action="store",
        type=int,
        nargs="*",
        default=None,
        help="Query index of the query that needs to be executed",
    )

    return parser


# For executable script
if __name__ == "__main__":

    parser = create_parser()
    args = parser.parse_args()

    config_fname_outer_scope = args.config_fname
    q_idx_outer_scope = args.q_idx

    main(config_fname_outer_scope, q_idx_outer_scope)
