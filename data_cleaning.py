def clean_df(df):
    import numpy as np
    import pandas as pd
    with pd.option_context('mode.use_inf_as_na', True):
        df = df.dropna()
    is_good_mom = (df['pz'] > 0) & (df['p']<20) & (df['pT']<3)
    is_good_coord = (abs(df['x']) < 100) & (abs(df['y']) < 100) & (df['z']>0) & (df['z']<80)
    is_good_params = (df['distance'] > 0) & (df['distance'] < 100) & (df['chi2geo']>0) & (df['chi2geo'] < 1000) & (df['cosinepos'] > 0.5) & (df['chi2topo'] > 0) & (df['chi2topo'] < 100000) & (df['cosineneg']>0.1) & (df['eta']>1) & (df['eta']<6.5) & (df['l']<80) & (df['ldl']>0) & (df['ldl']<5000)
    is_good_daughters = (df['chi2primneg']>0) & (df['chi2primneg'] < 3e7) & (df['chi2primpos']>0) & (df['chi2primpos']<1e6)
    is_good_mass = (df['mass']>1.1) & (df['mass']<2)

    is_good_df = (is_good_mom) & (is_good_coord) & (is_good_params) & (is_good_daughters) & (is_good_mass)

    return df[is_good_df]
