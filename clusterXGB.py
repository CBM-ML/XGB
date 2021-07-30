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


signal_path = '/home/olha/CBM/dataset10k_tree/dcm_1m_prim_signal.root'
df_urqmd_path = '/home/olha/CBM/dataset10k_tree/urqmd_100k_cleaned.root'

tree_name = 'PlainTree'

signal= tree_importer(signal_path,tree_name, 3)
df_urqmd = tree_importer(df_urqmd_path, tree_name,3)

background_selected = df_urqmd[(df_urqmd['issignal'] == 0) &\
                             ((df_urqmd['mass'] > 1.07) &\
                             (df_urqmd['mass'] < 1.108) | (df_urqmd['mass']>1.1227) &\
                             (df_urqmd['mass'] < 1.3))]


#Let's combine signal and background
df_scaled = pd.concat([signal, background_selected])
# Let's shuffle the rows randomly
df_scaled = df_scaled.sample(frac=1)
# Let's take a look at the top 10 entries of the df
df_scaled.iloc[0:10,:]
del signal, background_selected


# The following columns will be used to predict whether a reconstructed candidate is a lambda particle or not
# cuts = [ 'chi2primneg', 'chi2primpos', 'ldl', 'distance', 'chi2geo']

cuts = [ 'chi2primneg', 'chi2primpos']

x = df_scaled[cuts].copy()

# The MC information is saved in this y variable
y =pd.DataFrame(df_scaled['issignal'], dtype='int')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=324)

#DMatrix is a internal data structure that used by XGBoost which is optimized for both memory efficiency and training speed.
dtrain = xgb.DMatrix(x_train, label = y_train)
dtest1=xgb.DMatrix(x_test, label = y_test)


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


#Invoking the Bayesian Optimizer with the specified parameters to tune
xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (4, 10),
                                             'gamma': (0, 1),
                                            'alpha': (2,20),
                                             'learning_rate':(0,1),
                                             'n_estimators':(100,500)
                                            })


#performing Bayesian optimization for 5 iterations with 8 steps of random exploration with an #acquisition function of expected improvement
xgb_bo.maximize(n_iter=5, init_points=5)


max_param = xgb_bo.max['params']
param= {'alpha': max_param['alpha'], 'gamma': max_param['gamma'], 'learning_rate': max_param['learning_rate'], 'max_depth': int(round(max_param['max_depth'],0)), 'n_estimators': int(round(max_param['n_estimators'],0)), 'objective': 'reg:logistic'}
gc.collect()


#To train the algorithm using the parameters selected by bayesian optimization
#Fit/train on training data
bst = xgb.train(param, dtrain)

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
#ax.figure.savefig("hits.png")

#ROC cures for the predictions on train and test sets
train_best, test_best = plot_tools.AMS(y_train, bst_train['xgb_preds'],y_test, bst_test['xgb_preds'])

#The first argument should be a data frame, the second a column in it, in the form 'preds'
plot_tools.preds_prob(bst_test,'xgb_preds', 'issignal','test')

#To save some memory on colab we delete some unused variables
del dtrain, dtest1, x_train, x_test, y_train, y_test, df_scaled
gc.collect()


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


#lets take the best threshold and look at the confusion matrix
cut1 = test_best
df_dcm['xgb_preds1'] = ((df_dcm['xgb_preds']>cut1)*1)
cnf_matrix = confusion_matrix(df_dcm['issignal'], df_dcm['xgb_preds1'], labels=[1,0])
np.set_printoptions(precision=2)
fig, axs = plt.subplots(figsize=(10, 8))
axs.yaxis.set_label_coords(-0.04,.5)
axs.xaxis.set_label_coords(0.5,-.005)
plot_tools.plot_confusion_matrix(cnf_matrix, classes=['signal','background'], title='Confusion Matrix for XGB for cut > '+str(cut1))
#plt.savefig('confusion_matrix_extreme_gradient_boosting_whole_data.png')


plot_tools.cut_visualization(df_urqmd,'xgb_preds',test_best)

xgb.to_graphviz(bst, fmap='', num_trees=0, rankdir=None, yes_color=None, no_color=None, condition_node_params=None, leaf_node_params=None)

new_check_set= df_urqmd.copy()
new_check_set['new_signal']=0
mask1 = (new_check_set['chi2primpos'] > 18.4) & (new_check_set['chi2primneg'] > 18.4)

mask2 = (new_check_set['ldl'] > 5) & (new_check_set['distance'] < 1)

mask3 = (new_check_set['chi2geo'] < 3)

new_check_set = new_check_set[(mask1) & (mask2) & (mask3)]

#After all these cuts, what is left is considered as signal, so we replace all the values in the 'new_signal'
# column by 1
new_check_set['new_signal'] = 1
cnf_matrix = confusion_matrix(new_check_set['issignal'], new_check_set['new_signal'], labels=[1,0])
np.set_printoptions(precision=2)
fig, axs = plt.subplots(figsize=(10, 8))
axs.yaxis.set_label_coords(-0.04,.5)
axs.xaxis.set_label_coords(0.5,-.005)
plot_tools.plot_confusion_matrix(cnf_matrix, classes=['signal','background'], title='Confusion Matrix for KFPF')

cut3 = test_best
mask1 = df_original['xgb_preds']>cut3
df3=df_original[mask1]

plot_tools.comaprison_XGB_KFPF(df3['mass'],new_check_set['mass'])

del x,y,x_test,y_test,x_whole,y_whole,dtest,dtrain,dtest1,df3,df_clean,df_scaled
gc.collect()
