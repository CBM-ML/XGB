import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from library.CBM_ML.tree_importer import tree_importer

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz

from sklearn.model_selection import cross_val_score
from scipy.stats import uniform

from numpy import sqrt, log, argmax

from bayes_opt import BayesianOptimization

import gc

# Put here your path and your tree
signal_path = '/home/olha/CBM/dataset10k_tree/dcm_1m_prim_signal.root'
df_urqmd_path = '/home/olha/CBM/dataset10k_tree/urqmd_100k_cleaned.root'
df_dcm_path = '/home/olha/CBM/dataset10k_tree/dcm_prim_100k_cleaned.root'

tree_name = 'PlainTree'


# How many threads we use to parallel code
number_of_threads = 3


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

    background_selected = df_urqmd[(df_urqmd['issignal'] == 0) &\
                                 ((df_urqmd['mass'] > 1.07) &\
                                 (df_urqmd['mass'] < 1.108) | (df_urqmd['mass']>1.1227) &\
                                 (df_urqmd['mass'] < 1.3))]

    df_scaled = pd.concat([signal, background_selected])
    df_scaled = df_scaled.sample(frac=1)
    df_scaled.iloc[0:10,:]
    del signal, background_selected

    return df_scaled


df_scaled = data_selection(signal_path, df_dcm_path, tree_name,
 number_of_threads)
df_dcm = tree_importer(df_dcm, tree_name, number_of_threads)
df_urqmd = tree_importer(df_urqmd_path, tree_name, number_of_threads)
cuts = [ 'chi2primneg', 'chi2primpos']


def train_test_set(df_scaled, cuts):
    """
    To make machine learning algorithms more efficient on unseen data we divide
    our data into two sets. One set is for training the algorithm and the other
    is for testing the algorithm. If we don't do this then the algorithm can
    overfit and we will not capture the general trends in the data.

    Parameters
    ----------
    df_scaled: dataframe
          dataframe with mixed signal and bacjground
    cuts: list(contains strings)
          features on which training is based
    """
    x = df_scaled[cuts].copy()

    # The MC information is saved in this y variable
    y =pd.DataFrame(df_scaled['issignal'], dtype='int')
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=324)

    #DMatrix is a internal data structure that used by XGBoost
    # which is optimized for both memory efficiency and training speed.
    dtrain = xgb.DMatrix(x_train, label = y_train)
    dtest1=xgb.DMatrix(x_test, label = y_test)

    return dtrain, dtest1


dtrain, dtest = train_test_set(df_scaled, cuts)

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
plt.rcParams['figure.figsize'] = [5, 3]
plt.show()
ax.figure.tight_layout()
# ax.figure.savefig("hits.png")




#ROC cures for the predictions on train and test sets
train_best, test_best = plot_tools.AMS(y_train, bst_train['xgb_preds'],y_test, bst_test['xgb_preds'])

#The first argument should be a data frame, the second a column in it, in the form 'preds'
plot_tools.preds_prob(bst_test,'xgb_preds', 'issignal','test')

#To save some memory on colab we delete some unused variables
del dtrain, dtest1, x_train, x_test, y_train, y_test, df_scaled
gc.collect()




def whole_dataset(df_urqmd,df_dcm, cuts):
    """
    Makes predictions and adds it to dataset. df_dcm is dataset which wasn't used
    by model, so let's have a look what predictions we have for real data

    Parameters
    ----------
    df_urqmd: dataframe
          100k events data set
    df_dcm: dataframe
          100k events data set(unknown data)
    cuts: list(contains strings)
          features on which training is based

    """

    x_whole_1 = df_urqmd[cuts].copy()
    y_whole_1 = pd.DataFrame(df_urqmd['issignal'], dtype='int')
    dtest2 = xgb.DMatrix(x_whole_1, label = y_whole_1)
    df_urqmd['xgb_preds'] = bst.predict(dtest2, output_margin=False)

    del x_whole_1, y_whole_1, dtest2
    gc.collect()

    x_whole = df_dcm[cuts].copy()
    y_whole = pd.DataFrame(df_dcm['issignal'], dtype='int')
    #DMatrix is a internal data structure that used by XGBoost which is optimized for both memory efficiency and training speed.
    dtest = xgb.DMatrix(x_whole, label = y_whole)
    del x_whole, y_whole
    df_dcm['xgb_preds'] = bst.predict(dtest, output_margin=False)
    del dtest
    gc.collect()

    return df_urqmd,df_dcm



whole_dataset(df_urqmd,df_dcm, cuts)




def CM_plot(test_best, df_dcm):
    """
    Plots confusion matrix. A Confusion Matrix C is such that Cij is equal to
    the number of observations known to be in group i and predicted to be in
    group j. Thus in binary classification, the count of true positives is C00,
    false negatives C01,false positives is C10, and true neagtives is C11.

    Confusion matrix is applied to previously unseen by model data, so we can
    estimate model's performance

    Parameters
    ----------
    test_best:

    df_dcm: dataframe
          100k events data set(unknown data)
    """
    #lets take the best threshold and look at the confusion matrix
    cut1 = test_best
    df_dcm['xgb_preds1'] = ((df_dcm['xgb_preds']>cut1)*1)
    cnf_matrix = confusion_matrix(df_dcm['issignal'], df_dcm['xgb_preds1'], labels=[1,0])
    np.set_printoptions(precision=2)
    fig, axs = plt.subplots(figsize=(10, 8))
    axs.yaxis.set_label_coords(-0.04,.5)
    axs.xaxis.set_label_coords(0.5,-.005)
    plot_tools.plot_confusion_matrix(cnf_matrix, classes=['signal','background'],
     title='Confusion Matrix for XGB for cut > '+str(cut1))
    #plt.savefig('confusion_matrix_extreme_gradient_boosting_whole_data.png')


CM_plot(test_best, df_dcm)
