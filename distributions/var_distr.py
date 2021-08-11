#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gc
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties

import matplotlib as mpl


mpl.rc('figure', max_open_warning = 0)

def hist_variables(dfs_orig, dfb_orig, dfs_cut, dfb_cut,feature, pdf_key):
    """
    Applied quality cuts and created distributions for all the features in pdf
    file
    Parameters
    ----------
    df_s: dataframe
          signal
    df_b: dataframe
          background
    feature: str
            name of the feature to be plotted
    pdf_key: PdfPages object
            name of pdf document with distributions
    """

    # fig, ax = plt.subplots(figsize=(20, 10))

    fig, ax = plt.subplots(2, figsize=(20, 10))


    fontP = FontProperties()
    fontP.set_size('xx-large')

    ax[0].hist(dfs_orig[feature], label = 'signal', bins = 500, alpha = 0.4, color = 'blue')
    ax[0].hist(dfb_orig[feature], label = 'background', bins = 500, alpha = 0.4, color = 'red')
    ax[0].legend(shadow=True,title = 'S/B='+ str(round(len(dfs_orig)/len(dfb_orig), 3)) + '\n inf, nan was deleted \n $\chi^2$>0 '+
              '\n mass > 1.077 Gev/c , pz >0'+
               '\n z > 0, z<80, l > 0, l < 80, ldl > 0, |x|,|y|<50'+
               '\n cosinepos, cosineneg > 0' +
               '\n distance > 0, distance <100'
               '\n S samples:  '+str(dfs_orig.shape[0]) + '\n B samples: '+ str(dfb_orig.shape[0])
               , title_fontsize=20, fontsize =20, bbox_to_anchor=(1.05, 1),
                loc='upper left', prop=fontP,)


    ax[0].xaxis.set_tick_params(labelsize=25)
    ax[0].yaxis.set_tick_params(labelsize=25)

    ax[0].set_title(str(feature) + ' MC ', fontsize = 25)
    ax[0].set_xlabel(feature, fontsize = 25)


    ax[0].set_yscale('log')

    fig.tight_layout()


    ax[1].hist(dfs_cut[feature], label = 'signal', bins = 500, alpha = 0.4, color = 'blue')
    ax[1].hist(dfb_cut[feature], label = 'background', bins = 500, alpha = 0.4, color = 'red')
    ax[1].legend(shadow=True,title = 'S/B='+ str(round(len(dfs_cut)/len(dfb_cut), 3)) + '\n inf, nan was deleted \n $\chi^2$>0 '+
              '\n mass > 1.077 Gev/c , pz >0'+
               '\n z > 0, z<80, l > 0, l < 80, ldl > 0, |x|,|y|<50'+
               '\n cosinepos, cosineneg > 0' +
               '\n distance > 0, distance <100'
               '\n S samples:  '+str(dfs_cut.shape[0]) + '\n B samples: '+ str(dfb_cut.shape[0]) +
               '\n ML cut'
               , title_fontsize=20, fontsize =20, bbox_to_anchor=(1.05, 1),
                loc='upper left', prop=fontP,)


    ax[1].xaxis.set_tick_params(labelsize=25)
    ax[1].yaxis.set_tick_params(labelsize=25)

    ax[1].set_title(feature + ' MC ', fontsize = 25)
    ax[1].set_xlabel(feature, fontsize = 25)


    ax[1].set_yscale('log')

    fig.tight_layout()

    plt.savefig(pdf_key,format='pdf')
