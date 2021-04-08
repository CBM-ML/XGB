"""
In the already existing Kalman Filter Particle Finder (KFPF) package for online reconstruction and selection of short-lived particles in CBM, these criteria
have been manually optimized. These selection-cuts have been selected to maximize the signal to background ratio (S/B) of the $\Lambda$ for a certain energy
on a collisions generator. The selection criteria mainly depends on the collision energy, decay channel and detector configuration.

The following function takes in a data-frame as an input and returns a data frame after the application of the selection criteria.
"""

def KFPF_lambda_cuts(df):
    KFPF_lambda= df.copy()
    KFPF_lambda['new_signal']=0
    mask1 = (KFPF_lambda['chi2primpos'] > 18.4) & (KFPF_lambda['chi2primneg'] > 18.4)

    mask2 = (KFPF_lambda['ldl'] > 5) & (KFPF_lambda['distance'] < 1)

    mask3 = (KFPF_lambda['chi2geo'] < 3) & (KFPF_lambda['cosinepos'] > 0) & (KFPF_lambda['cosineneg'] > 0)

    KFPF_lambda = KFPF_lambda[(mask1) & (mask2) & (mask3)] 

    #After all these cuts, what is left is considered as signal, so we replace all the values in the 'new_signal'
    # column by 1
    KFPF_lambda['new_signal'] = 1
    return KFPF_lambda
