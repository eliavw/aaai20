from aaai20.io import (
    build_filesystem,
    default_prefix_exp_fn_suffix,
    filename_config,
    filename_cli_commands,
    filename_script,
    filename_logs,
    exp_directory
)

from pathlib import Path
import pandas as pd
import os
from os.path import abspath


def prepare_fs(exp_keyword, script, fs):
    directory = exp_directory(exp_dname=exp_keyword, script=script, fs=fs)
    
    for subdir_name, subdir in directory.items():
        if subdir_name in {'script', 'current_exp'}:
            pass
        else:
            if not os.path.exists(subdir):
                os.makedirs(subdir)
    return


def generate_config_and_log_fname(config, kind='fit', qry_idx=None, exp_fn_fields=None, config_needs_suffix=False):
    
    if kind in {'fit'}:
        p, f, s = default_prefix_exp_fn_suffix(
                config, fit_config = config.get("fit_config"), qry_idx=qry_idx, exp_fn_fields= exp_fn_fields
            )
    elif kind in {'predict'}:
        p, f, s = default_prefix_exp_fn_suffix(
                config, predict_config = config.get("predict_config"), qry_idx=qry_idx, exp_fn_fields= exp_fn_fields
            )
    
    if config_needs_suffix:
        config_suffix = s
    else:
        config_suffix = ""

    fn_cfg = filename_config(
        prefix=p,
        suffix=config_suffix,
        exp_fname=f,
        exp_dname=config.get("exp_keyword"),
        script=config.get("script"),
        extension='json'
    )
    
    fn_log = filename_logs(
        prefix=p,
        suffix=s,
        exp_fname=f,
        extension="",
        exp_dname=config.get("exp_keyword"),
        script=config.get("script"),
    )
    return fn_cfg, fn_log


def generate_df_commands(fn_script, fn_cfg, fn_log, timeout, q_idx=None, shuffle=True):
    df = pd.DataFrame()
    
    fn_log = [abspath(f) for f in fn_log]
    fn_cfg = [abspath(f) for f in fn_cfg]
    fn_script = abspath(fn_script)
    
    
    df['log_fname'] = fn_log
    df['config_fname'] = fn_cfg
    df['script_fname'] = fn_script
    df['timeout'] =  timeout
    
    df = df[['script_fname', 'config_fname', 'log_fname', 'timeout']]
    
    if q_idx is not None:
        df["q_idx"] = q_idx
        
    if shuffle:
        # Shuffle all commands, cf. https://stackoverflow.com/questions/29576430/shuffle-dataframe-rows
        df = df.sample(frac=1).reset_index(drop=True)
        
    return df


# Redundant!
def _add_queries_in_cartesian_product_way(df_original, queries_list):
    df_tmp = pd.DataFrame()
    df_tmp['q_idx'] = queries_list

    df_tmp['key'] = 'DUMMY-KEY'
    df_original['key'] = 'DUMMY-KEY'

    df_cartesian = df_original.merge(df_tmp, how='outer')
    df_cartesian = df_cartesian.drop(columns=['key'])

    return df_cartesian


# For determination of start_idx
def all_fnames_in_dir(dname):
    paths = list(Path(dname).rglob("*.[cC][sS][vV]"))
    fnames = [p.stem for p in paths]
    return fnames


def extract_idx_from_fnames(fnames):
    # This is basically hardcoded to detect exp_idx!
    idxs = [int(f.split("-")[0]) for f in fnames]
    idxs.sort()
    return idxs


def default_start_idx(fs=None, script="run_mercs"):
    if fs is None:
        fs = build_filesystem()
    
    dname = fs[script]
    fnames = all_fnames_in_dir(dname)
    idxs = extract_idx_from_fnames(fnames)
    try:
        return max(idxs) +1
    except:
        print("Nothing found, so index is 0")
        return 0