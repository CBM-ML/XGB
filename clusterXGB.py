import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from library.CBM_ML.tree_importer import tree_importer, new_labels, quality_cuts
from library.CBM_ML.plot_tools import AMS, preds_prob,plot_confusion_matrix

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz

from sklearn.model_selection import cross_val_score
from scipy.stats import uniform

from numpy import sqrt, log, argmax

from bayes_opt import BayesianOptimization

import gc

import sys
import os

from distributions.var_distr import hist_variables
from matplotlib.backends.backend_pdf import PdfPages


tree_name = 'PlainTree'

path_list = []

# How many threads we use to parallel code
number_of_threads = 3

for x in sys.argv[1:]:
    path_list.append(x)

signal_path = path_list[0]
df_urqmd_path = path_list[1]

output_path = path_list[2]

if not os.path.exists(output_path):
    os.mkdir(output_path)

def data_selection(signal_path, bgr_path, tree, threads):

    """
    We have selected signal candidates only from the DCM model, therefore, we are
    treating it as simulated data. The URQMD data set will be treated as real
    experimental data.

    Our URQMD 100k events data, which looks more like what we will get from the
    final experiment, has a lot more background than signal. This problem of
    unequal ratio of classes (signal and background) in our data set (URQMD,
     99.99% background and less than 1% signal) is called imbalance classification problem.

    One of the solutions to tackle this problem is resampling the data.
    Deleting instances from the over-represented class (in our case the background),
    under-sampling, is a resampling method.

    So for training and testing we will get signal candidates from the DCM signal
    and background from URQMD (3 times signal size).

    Parameters
    ----------
    signal_path: str
          path to signal file
    bgr_path: str
          path to background file
    tree: str
          name of flat tree
    threads: int
            how many parallel thread we want
    """
    signal= tree_importer(signal_path,tree_name, threads)
    df_urqmd = tree_importer(bgr_path, tree_name, threads)

    signal = new_labels(signal)
    df_urqmd = new_labels(df_urqmd)

    signal = quality_cuts(signal)
    df_urqmd = quality_cuts(df_urqmd)

    signal_selected = signal[signal['issignal']==1]
    background_selected = df_urqmd[(df_urqmd['issignal'] == 0) &\
                                 ((df_urqmd['mass'] > 1.07) &\
                                 (df_urqmd['mass'] < 1.108) | (df_urqmd['mass']>1.1227) &\
                                 (df_urqmd['mass'] < 1.3))]

    df_scaled = pd.concat([signal_selected, background_selected])
    df_scaled = df_scaled.sample(frac=1)
    df_scaled.iloc[0:10,:]
    del signal, background_selected

    return df_scaled


df_scaled = data_selection(signal_path, df_urqmd_path, tree_name,
 number_of_threads)


# features to be trained
cuts = [ 'chi2primneg', 'chi2primpos','chi2geo','distance', 'ldl']


def train_test_set(df_scaled, cuts):
    """
    To make machine learning algorithms more efficient on unseen data we divide
    our data into two sets. One set is for training the algorithm and the other
    is for testing the algorithm. If we don't do this then the algorithm can
    overfit and we will not capture the general trends in the data.

    Parameters
    ----------
    df_scaled: dataframe
          dataframe with mixed signal and background
    cuts: list(contains strings)
          features on which training is based
    """
    # x = df_scaled[cuts].copy()
    x = df_scaled.copy()

    # The MC information is saved in this y variable
    y =pd.DataFrame(df_scaled['issignal'], dtype='int')
    x_train_all, x_test_all, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=324)


    x_train = x_train_all[cuts].copy()
    x_test = x_test_all[cuts].copy()
    #DMatrix is a internal data structure that used by XGBoost
    # which is optimized for both memory efficiency and training speed.
    dtrain = xgb.DMatrix(x_train, label = y_train)
    dtest1=xgb.DMatrix(x_test, label = y_test)

    return dtrain, dtest1,x_train_all,x_test_all,x_train, x_test, y_train, y_test


dtrain, dtest1,x_train_all, x_test_all, x_train,x_test, y_train, y_test = train_test_set(df_scaled, cuts)

del df_scaled
gc.collect()




#Bayesian Optimization function for xgboost
#specify the parameters you want to tune as keyword arguments
def bo_tune_xgb(max_depth, gamma, alpha, n_estimators ,learning_rate):
    params = {'max_depth': int(max_depth),
              'gamma': gamma,
              'alpha':alpha,
              'n_estimators': n_estimators,
              'learning_rate':learning_rate,
              'subsample': 0.8,
              'eta': 0.3,
              'eval_metric': 'auc', 'nthread' : 7}
    cv_result = xgb.cv(params, dtrain, num_boost_round=10, nfold=5)
    return  cv_result['test-auc-mean'].iloc[-1]



def get_best_params():
    """
    Performs Bayesian Optimization and looks for the best parameters

    Parameters:
           None
    """
    #Invoking the Bayesian Optimizer with the specified parameters to tune
    xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (4, 10),
                                                 'gamma': (0, 1),
                                                'alpha': (2,20),
                                                 'learning_rate':(0,1),
                                                 'n_estimators':(100,500)
                                                })
    #performing Bayesian optimization for 5 iterations with 8 steps of random exploration
    # with an #acquisition function of expected improvement
    xgb_bo.maximize(n_iter=1, init_points=1)

    max_param = xgb_bo.max['params']
    param= {'alpha': max_param['alpha'], 'gamma': max_param['gamma'], 'learning_rate': max_param['learning_rate'],
     'max_depth': int(round(max_param['max_depth'],0)), 'n_estimators': int(round(max_param['n_estimators'],0)),
      'objective': 'reg:logistic'}
    gc.collect()


    #To train the algorithm using the parameters selected by bayesian optimization
    #Fit/train on training data
    bst = xgb.train(param, dtrain)
    return bst


bst = get_best_params()

#predicitions on training set
bst_train= pd.DataFrame(data=bst.predict(dtrain, output_margin=False),  columns=["xgb_preds"])
y_train=y_train.set_index(np.arange(0,bst_train.shape[0]))
bst_train['issignal']=y_train['issignal']


#predictions on test set
bst_test = pd.DataFrame(data=bst.predict(dtest1, output_margin=False),  columns=["xgb_preds"])
y_test=y_test.set_index(np.arange(0,bst_test.shape[0]))
bst_test['issignal']=y_test['issignal']


#The following graph will show us that which features are important for the model
ax = xgb.plot_importance(bst)
plt.rcParams['figure.figsize'] = [6, 3]
plt.show()
ax.figure.tight_layout()
ax.figure.savefig(str(output_path)+"/hits.png")


#ROC cures for the predictions on train and test sets
train_best, test_best = AMS(y_train, bst_train['xgb_preds'],y_test, bst_test['xgb_preds'], output_path)

#The first argument should be a data frame, the second a column in it, in the form 'preds'
preds_prob(bst_test,'xgb_preds', 'issignal','test', output_path)


def CM_plot(best, x, output_path):
    """
    Plots confusion matrix. A Confusion Matrix C is such that Cij is equal to
    the number of observations known to be in group i and predicted to be in
    group j. Thus in binary classification, the count of true positives is C00,
    false negatives C01,false positives is C10, and true neagtives is C11.

    Confusion matrix is applied to previously unseen by model data, so we can
    estimate model's performance

    Parameters
    ----------
    test_best: numpy.float32
              best threshold

    x_train: dataframe
            we want to get confusion matrix on training datasets
    """
    #lets take the best threshold and look at the confusion matrix
    cut1 = best
    x['xgb_preds1'] = ((x['xgb_preds']>cut1)*1)
    cnf_matrix = confusion_matrix(x['issignal'], x['xgb_preds1'], labels=[1,0])
    np.set_printoptions(precision=2)
    fig, axs = plt.subplots(figsize=(10, 8))
    axs.yaxis.set_label_coords(-0.04,.5)
    axs.xaxis.set_label_coords(0.5,-.005)
    plot_confusion_matrix(cnf_matrix, classes=['signal','background'],
     title='Confusion Matrix for XGB for cut > '+str(cut1))
    plt.savefig(str(output_path)+'/confusion_matrix_extreme_gradient_boosting_whole_data.png')


CM_plot(test_best, bst_test, output_path)

print("x_train_all: ", len(x_train_all))
print("x_test_all: ", len(x_test_all))

x_test_all['issignalXGB'] = bst_test['xgb_preds'].values
x_test_all['xgb_preds1'] = ((x_test_all['issignalXGB']>test_best)*1)

x_test_all['issignal'] = y_test.values

dfs_orig = x_test_all[x_test_all['issignal']==1]
dfb_orig = x_test_all[x_test_all['issignal']==0]


dfs_cut = x_test_all[(x_test_all['xgb_preds1']==1) & (x_test_all['issignal']==1)]
dfb_cut = x_test_all[(x_test_all['xgb_preds1']==1) & (x_test_all['issignal']==0)]

difference_s = pd.concat([dfs_orig, dfs_cut]).drop_duplicates(keep=False)

print("x_test_all: ", len(x_test_all))
print("dfs_orig: ", len(dfs_orig))

print("dfs_cut: ", len(dfs_cut))
print("difference: ", len(difference_s))

non_log_x = ['cosineneg', 'cosinepos', 'cosinetopo',  'mass', 'pT', 'rapidity',
 'phi', 'eta', 'x', 'y','z', 'px', 'py', 'pz', 'l', 'ldl']

log_x = ['chi2geo', 'chi2primneg', 'chi2primpos', 'chi2topo', 'distance']

new_log_x = []

cuts1 = ['chi2geo', 'chi2primneg', 'chi2primpos', 'chi2topo', 'cosineneg',
   'cosinepos', 'cosinetopo', 'distance', 'eta', 'l', 'ldl',
   'mass', 'p', 'pT', 'phi', 'px', 'py', 'pz', 'rapidity', 'x', 'y', 'z']

for cut in cuts1:
    if cut in log_x:
        dfs_orig[cut+'_log'] = np.log(dfs_orig[cut])
        dfb_orig[cut+'_log'] = np.log(dfb_orig[cut])

        dfs_cut[cut+'_log'] = np.log(dfs_cut[cut])
        dfb_cut[cut+'_log'] = np.log(dfb_cut[cut])

        difference_s[cut+'_log'] = np.log(difference_s[cut])

        new_log_x.append(cut+'_log')


        dfs_orig = dfs_orig.drop([cut], axis=1)
        dfb_orig = dfb_orig.drop([cut], axis=1)

        dfs_cut = dfs_cut.drop([cut], axis=1)
        dfb_cut = dfb_cut.drop([cut], axis=1)
        difference_s = difference_s.drop([cut], axis=1)

    if cut in non_log_x:
        new_log_x.append(cut)


pdf_cuts = PdfPages(output_path+'/'+'dist_cuts.pdf')
for feat in new_log_x:
    hist_variables(dfs_orig, dfb_orig, dfs_cut, dfb_cut, difference_s, feat, pdf_cuts)

pdf_cuts.close()
