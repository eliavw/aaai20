import numpy as np
import pandas as pd
from modulo.utils.encoding import encode_attribute
from .io import filename_results


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


def collect_results(dataset, q_codes, results, algorithm='weka'):
    df = pd.DataFrame()
    
    n_qry, n_att = q_codes.shape
    miss_encoding = encode_attribute(2, [0], [1])
    n_miss = np.sum((q_codes == miss_encoding), axis=1)
    
    
    df['q_idx'] = range(n_qry)
    df['dataset'] = dataset
    df['F1'] = results
    df['missing_percentage'] = n_miss/n_att
    df['difficulty'] = _convert_percentage_missing_to_difficulty(df['missing_percentage'].values)
    df['algorithm'] = algorithm
    return df

def _convert_percentage_missing_to_difficulty(percentage_missing):
    return np.ceil(percentage_missing*10).astype(int)
