import uproot
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd


"""
The tree_importer function takes in flat analysis tree and returns a pandas data-frame object. It has 3 inputs, the first one is
path of the analysis tree, the second one as the tree name and third one the number of CPU cores. The first and second input
should be inserted as strings i.e. path inside a single quotation '' or double quotations "". The third input should be a number.
For example  tree_importer("/home/flat_trees/a.tree","PlainTree",4)
"""
def tree_importer(path,treename, n):
    #The number of parallel processors
    executor = ThreadPoolExecutor(n)

    #To open the root file and convert it to a pandas dataframe
    file = uproot.open(path+':'+treename, library='pd', decompression_executor=executor,
                                  interpretation_executor=executor).arrays(library='np',decompression_executor=executor,
                                  interpretation_executor=executor)
    df= pd.DataFrame(data=file)
    return df


def new_labels(df):
    new_labels= ['chi2geo', 'chi2primneg', 'chi2primpos', 'chi2topo', 'cosineneg',
       'cosinepos', 'cosinetopo', 'distance', 'eta', 'l', 'ldl',
       'mass', 'p', 'pT', 'phi', 'px', 'py', 'pz', 'rapidity', 'x', 'y', 'z',
       'daughter1id', 'daughter2id', 'isfrompv', 'pid', 'issignal']

    df.columns = new_labels
    return df

def quality_cuts(df):

    """
     All the numerical artifacts ( $\chi$2 < 0), inf and nan were deleted. Also applied
     quality cuts based on detector geometry. Full description could be found in
     https://docs.google.com/document/d/11f0ZKPW8ftTVhTxeWiog1g6qdsGgN1mlIE3vd5FHLbc/edit?usp=sharing
     Parameters
     ------------------
     df: dataframe
         dataframe to be cleaned
    """

    with pd.option_context('mode.use_inf_as_na', True):
        df = df.dropna()

    df = df.dropna()

    chi2_cut = (df['chi2geo'] > 0) & (df['chi2primpos'] > 0) & (df['chi2primneg'] > 0) &\
           (df['chi2topo'] > 0)
    mass_cut = (df['mass'] > 1.077)

    coord_cut = (abs(df['x']) < 50) & (abs(df['y']) < 50)
    dist_l_cut = (df['distance'] > 0) &  (df['distance'] < 100) &\
                     (df['l'] > 0 )  & (df['ldl'] > 0 ) & (abs(df['l']) < 80)

    pz_cut = (df['pz'] > 0)

    cuts = (chi2_cut) & (mass_cut) & (coord_cut) & (dist_l_cut) &\
    (pz_cut)

    return df[cuts]
