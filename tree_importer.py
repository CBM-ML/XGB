import uproot
from concurrent.futures import ThreadPoolExecutor
"""
The tree_importer function takes in flat analysis tree and returns a pandas data-frame object. The executor variable accepts the number of parallel processors
available.
"""

def tree_importer(path,treename):
    import pandas as pd
    executor = ThreadPoolExecutor(8)
    file = uproot.open(path+':'+treename+'', library='pd', decompression_executor=executor,
                                  interpretation_executor=executor).arrays(library='np',decompression_executor=executor,
                                  interpretation_executor=executor)
    df= pd.DataFrame(data=file)
    return df
