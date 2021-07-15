import uproot
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
"""
The tree_importer function takes in flat analysis tree and returns a pandas data-frame object. The executor variable accepts the number of parallel processors
available.
"""

def tree_importer(path,treename, n):
    labels=["LambdaCandidates_chi2geo", "LambdaCandidates_chi2primneg", "LambdaCandidates_chi2primpos",
         "LambdaCandidates_distance", "LambdaCandidates_ldl","LambdaCandidates_mass", "LambdaCandidates_pT", "LambdaCandidates_rapidity", "LambdaCandidates_is_signal"]

    new_labels=['chi2geo', 'chi2primneg','chi2primpos', 'distance', 'ldl','mass', 'pT', 'rapidity','issignal']

    executor = ThreadPoolExecutor(n)
    file = uproot.open(path+':'+treename+'', library='pd', decompression_executor=executor,
                                  interpretation_executor=executor).arrays(labels,"(LambdaCandidates_mass < 1.3) & (LambdaCandidates_mass > 1.07) & (LambdaCandidates_pz>0) & (LambdaCandidates_p<20)  & (LambdaCandidates_distance>0) & (LambdaCandidates_distance<100) & (LambdaCandidates_chi2geo>0) & (LambdaCandidates_chi2geo<1000) & (LambdaCandidates_chi2topo>0) & (LambdaCandidates_chi2primpos<1e6) & (LambdaCandidates_chi2primneg < 3e7) & (LambdaCandidates_ldl>0) & (LambdaCandidates_ldl<5000) & (LambdaCandidates_chi2topo < 100000) & (LambdaCandidates_cosineneg>0.1) & (LambdaCandidates_cosinepos>0.1) & (LambdaCandidates_eta>1) & (LambdaCandidates_eta<6.5) & (LambdaCandidates_l<80) & (LambdaCandidates_x>-50) & (LambdaCandidates_x<50) & (LambdaCandidates_y>-50) & (LambdaCandidates_y<50) & (LambdaCandidates_z>-1) & (LambdaCandidates_z<80)", library='np',decompression_executor=executor,
                                  interpretation_executor=executor)
    df= pd.DataFrame(data=file)
    df.columns = new_labels
    #df['issignal']=((df['issignal']>0)*1)
    with pd.option_context('mode.use_inf_as_na', True):
        df = df.dropna()
    return df

