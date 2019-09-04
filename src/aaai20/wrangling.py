import arff
import pandas as pd

def arff_to_df(filename, encode_nominal=False, return_af=True):
    with open(filename, 'r') as f:
        af = arff.load(f, encode_nominal=encode_nominal)
    
    df = pd.DataFrame(af['data'])
    
    if return_af:
        return df, af
    else:
        return df