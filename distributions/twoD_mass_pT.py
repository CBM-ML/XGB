import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl

def plot2D(df, sample, sgn,x_axis_value, y_axis_value, range_x, range_y, pdf_key):
    """

    Plots 2-D histogram.

    x_axis_value: e.g. 'mass'
    y_axis_value: e.g. 'distance'
    range_x: e.g. [1, 1.177]
    range_y: e.g [0, 100]
    folder: e.g. 'folder'

    """

    fig, axs = plt.subplots(figsize=(15, 10))
    cax = plt.hist2d(df[x_axis_value],df[y_axis_value],range=[range_x, range_y], bins=100,
                norm=mpl.colors.LogNorm(), cmap=plt.cm.viridis)


    if x_axis_value=='mass':
        unit = r' $, \frac{GeV}{c^2}$ '
        plt.vlines(x=1.115683,ymin=range_y[0],ymax=range_y[1], color='r', linestyle='-')

    if x_axis_value=='pT':
        unit = r' $, \frac{GeV}{c}$'


    if sgn==1:
        plt.title('Signal candidates ' + sample, fontsize = 25)

    if sgn==0:
        plt.title('Background candidates' + sample, fontsize = 25)


    plt.xlabel(x_axis_value+unit, fontsize=25)
    plt.ylabel(y_axis_value, fontsize=25)



    mpl.pyplot.colorbar()

    plt.legend(shadow=True,title =str(len(df))+ " samples")

    fig.tight_layout()
    plt.savefig(pdf_key,format='pdf')
