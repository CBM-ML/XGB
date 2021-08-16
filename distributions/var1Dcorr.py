import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def vars(df, var_to_corr):
    vars_to_draw = list(df)
    vars_to_draw.remove(var_to_corr)
    return vars_to_draw


def calculate_correlation(df, vars_to_corr, target_var) :

    from scipy.stats import sem

    mean = df[target_var].mean()
    sigma = df[target_var].std()

    correlation = []
    error = []

    for j in vars_to_corr :
        mean_j = df[j].mean()
        sigma_j = df[j].std()

        cov = (df[j] - mean_j) * (df[target_var] - mean) / (sigma*sigma_j)
        correlation.append(cov.mean())
        error.append(sem(cov))

    return correlation, error


def plot1Dcorrelation(vars_to_draw,var_to_corr, corr_signal, corr_signal_errors, corr_bg, corr_bg_errors, output_path):
    fig, ax = plt.subplots(figsize=(20,10))
    plt.errorbar(vars_to_draw, corr_signal, yerr=corr_signal_errors, fmt='')
    plt.errorbar(vars_to_draw, corr_bg, yerr=corr_bg_errors, fmt='')
    ax.grid(zorder=0)
    ax.set_xticklabels(vars_to_draw, fontsize=25, rotation =70)
    ax.set_yticklabels([-0.5,-0.4,  -0.2,0, -0.2, 0.4], fontsize=25)
    plt.legend(('signal','background'), fontsize = 25)
    plt.title('Correlation of all variables with '+ var_to_corr+' along with SEM', fontsize = 25)
    plt.ylabel('Correlation coefficient', fontsize = 25)
    fig.tight_layout()
    fig.savefig(output_path+'/all_vars_corr-'+ var_to_corr+'.png')
