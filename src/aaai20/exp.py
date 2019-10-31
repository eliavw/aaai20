import numpy as np
import pandas as pd
from mercs.utils.encoding import encode_attribute
from .io import filename_results

from typing import List, Dict


def process_outcomes(dataframes):
    """
    Convert into standard-form dataframe.
    """
    df_list = [v for _, v in dataframes.items()]
    df = pd.concat(df_list)
    
    return df


def save_outcome(df, filename='weka'):
    fn = filename_results(filename)
    
    df.to_csv(fn)
    return


def collect_results(dataset, q_codes, results: Dict[str, List[float]], identifier='weka', q_idx: List[int]=None, **extra_fields):
    # Init
    df = pd.DataFrame()
    
    nb_qry, n_att = q_codes.shape
    miss_encoding = encode_attribute(2, [0], [1])
    n_miss = np.sum((q_codes == miss_encoding), axis=1)
    
    # Build DataFrame
    
    df['missing_percentage'] = n_miss/n_att
    df['difficulty'] = _convert_percentage_missing_to_difficulty(df['missing_percentage'].values)

    if q_idx is not None:
        df['q_idx'] = q_idx
    else:
        df['q_idx'] = range(nb_qry)

    for k in results:
        df[k] = results[k]

    df['identifier'] = identifier
    df['dataset'] = dataset

    for f in extra_fields:
        df[f] = extra_fields[f]

    return df


def collect_timings(timings, q_idx=None, **extra_fields):
    assert isinstance(q_idx, list)
    assert len(q_idx) == len(timings.get('inf_time'))
    
    df = pd.DataFrame()

    df["q_idx"] = q_idx

    for k in timings:
        df[k] = timings[k]

    for f in extra_fields:
        df[f] = extra_fields[f]

    return df


def _convert_percentage_missing_to_difficulty(percentage_missing):
    return np.ceil(percentage_missing*10).astype(int)
