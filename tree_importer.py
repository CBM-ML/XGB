import uproot
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
"""
The tree_importer function takes in flat analysis tree and returns a pandas data-frame object. The executor variable accepts the number of parallel processors
available.
"""

labels=["LambdaCandidates_chi2geo", "LambdaCandidates_chi2primneg", "LambdaCandidates_chi2primpos",
       "LambdaCandidates_chi2topo",
       "LambdaCandidates_cosineneg", "LambdaCandidates_cosinepos", "LambdaCandidates_distance",
       "LambdaCandidates_eta", "LambdaCandidates_l", "LambdaCandidates_ldl",
       "LambdaCandidates_mass", "LambdaCandidates_p", "LambdaCandidates_pT",
       "LambdaCandidates_phi", "LambdaCandidates_pz", "LambdaCandidates_rapidity",
       "LambdaCandidates_x", "LambdaCandidates_y", "LambdaCandidates_z","LambdaCandidates_is_signal"]

def tree_importer(path,treename):
    executor = ThreadPoolExecutor(8)
    file = uproot.open(path+':'+treename+'', library='pd', decompression_executor=executor,
                                  interpretation_executor=executor).arrays(labels,"(LambdaCandidates_mass < 2) & (LambdaCandidates_mass > 1.07)", library='np',decompression_executor=executor,
                                  interpretation_executor=executor)
    df= pd.DataFrame(data=file)
    return df
