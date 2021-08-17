import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from matplotlib.font_manager import FontProperties

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

import gc
import matplotlib as mpl



mpl.rc('figure', max_open_warning = 0)



def pT_vs_rapidity(df_orig, df_cut, difference, sign, x_range, y_range, output_path, data_name):
    fig, axs = plt.subplots(ncols=3, figsize=(15, 4))


    if sign ==0:
        s_label = 'Background'
        m = 5

    if sign==1:
        s_label = 'Signal'
        m = 1

    rej = round((1 -  (df_cut.shape[0] / df_orig.shape[0])) * 100, 5)
    diff = df_orig.shape[0] - df_cut.shape[0]
    axs[1].legend(shadow=True,title ='ML cut rejects \n'+ str(rej) +'% of '+ s_label,
     fontsize =15)

    counts0, xedges0, yedges0, im0 = axs[0].hist2d(df_orig['rapidity'], df_orig['pT'] , range = [x_range, y_range], bins=100,
                norm=mpl.colors.LogNorm(), cmap=plt.cm.rainbow)

    axs[0].set_title(s_label + ' candidates before ML cut '+data_name, fontsize = 16)
    axs[0].set_xlabel('rapidity', fontsize=15)
    axs[0].set_ylabel('pT, GeV', fontsize=15)


    mpl.pyplot.colorbar(im0, ax = axs[0])



    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[0].xaxis.set_major_formatter(FormatStrFormatter('%d'))

    axs[0].xaxis.set_tick_params(which='both', width=2)


    fig.tight_layout()


    counts1, xedges1, yedges1, im1 = axs[1].hist2d(df_cut['rapidity'], df_cut['pT'] , range = [x_range, y_range], bins=100,
                norm=mpl.colors.LogNorm(), cmap=plt.cm.rainbow)

    axs[1].set_title(s_label + ' candidates after ML cut '+data_name, fontsize = 16)
    axs[1].set_xlabel('rapidity', fontsize=15)
    axs[1].set_ylabel('pT, GeV', fontsize=15)

    mpl.pyplot.colorbar(im1, ax = axs[1])





    axs[1].xaxis.set_major_locator(MultipleLocator(1))
    axs[1].xaxis.set_major_formatter(FormatStrFormatter('%d'))

    axs[1].xaxis.set_tick_params(which='both', width=2)

    fig.tight_layout()


    counts2, xedges2, yedges2, im2 = axs[2].hist2d(difference['rapidity'], difference['pT'] , range = [x_range, y_range], bins=100,
                norm=mpl.colors.LogNorm(), cmap=plt.cm.rainbow)

    axs[2].set_title(s_label + ' candidates after ML cut '+data_name, fontsize = 16)
    axs[2].set_xlabel('rapidity', fontsize=15)
    axs[2].set_ylabel('pT, GeV', fontsize=15)

    mpl.pyplot.colorbar(im1, ax = axs[1])





    axs[2].xaxis.set_major_locator(MultipleLocator(1))
    axs[2].xaxis.set_major_formatter(FormatStrFormatter('%d'))

    axs[2].xaxis.set_tick_params(which='both', width=2)

    fig.tight_layout()

    fig.savefig(output_path+'/pT_rapidity_'+s_label+'_ML_cut_'+data_name+'.png')
