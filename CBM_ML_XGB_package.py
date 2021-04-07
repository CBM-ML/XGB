#We import some libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from scipy.stats import uniform
import weakref 
from bayes_opt import BayesianOptimization
from data_cleaning import clean_df
from KFPF_lambda_cuts import KFPF_lambda_cuts
from plot_tools import AMS, preds_prob, plot_confusion_matrix
import gc
from tree_importer import tree_importer

#To save some memory we will delete unused variables
class TestClass(object): 
    def check(self): 
        print ("object is alive!") 
    def __del__(self): 
        print ("object deleted") 
        
#to paralellize some part of the code
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(8)


# We import three root files into our jupyter notebook
signal= tree_importer('/home/shahid/cbmsoft/Data/PFSimplePlainTreeSignal.root','PlainTree')

# We only select lambda candidates
sgnal = signal[(signal['LambdaCandidates_is_signal']==1) & (signal['LambdaCandidates_mass']>1.108)
               & (signal['LambdaCandidates_mass']<1.1227)]

# Similarly for the background
background = tree_importer('/home/shahid/cbmsoft/Data/PFSimplePlainTreeBackground.root','PlainTree')
bg = background[(background['LambdaCandidates_is_signal'] == 0)
                & ((background['LambdaCandidates_mass'] > 1.07)
                & (background['LambdaCandidates_mass'] < 1.108) | (background['LambdaCandidates_mass']>1.1227) 
                   & (background['LambdaCandidates_mass'] < 2.00))]

#delete unused variables
del signal
del background

#we also import a 10k events data set, generated using URQMD with AuAu collisions at 12AGeV
file = tree_importer('/home/shahid/cbmsoft/Data/10k_events_PFSimplePlainTree.root','PlainTree')
#Call the python garbage collector to clean up things
gc.collect()
df_original= pd.DataFrame(data=file)
del file

#The labels of the columns in the df data frame are having the prefix LambdaCandidates_ so we rename them
new_labels= ['chi2geo', 'chi2primneg', 'chi2primpos', 'chi2topo', 'cosineneg',
       'cosinepos', 'cosinetopo', 'distance', 'eta', 'l', 'ldl',
       'mass', 'p', 'pT', 'phi', 'px', 'py', 'pz', 'rapidity',
             'x', 'y', 'z', 'daughter1id', 'daughter2id', 'isfrompv', 'pid', 'issignal']

sgnal.columns = new_labels
bg.columns = new_labels
df_original.columns=new_labels

# Next we clean the data using the clean_df function saved in another .py file

#Creating a new data frame and saving the results in it after cleaning of the original dfs
#Also keeping the original one
bcknd = clean_df(bg)
signal = clean_df(sgnal)

del bg
del sgnal
gc.collect()

df_clean = clean_df(df_original)
del df_original
gc.collect()

# We randomly choose our signal set of 4000 candidates
signal_selected= signal.sample(n=90000)

#background = 3 times the signal is also done randomly
background_selected = bcknd

del signal
del bcknd

#Let's combine signal and background
dfs = [signal_selected, background_selected]
df_scaled = pd.concat(dfs)
# Let's shuffle the rows randomly
df_scaled = df_scaled.sample(frac=1)
del dfs


# The following columns will be used to predict whether a reconstructed candidate is a lambda particle or not
cuts = [ 'chi2primneg', 'chi2primpos', 'ldl', 'distance', 'chi2geo']
x = df_scaled[cuts].copy()

# The MC information is saved in this y variable
y =pd.DataFrame(df_scaled['issignal'], dtype='int')

#We do the same for the 10k events data set
x_whole = df_clean[cuts].copy()
y_whole = pd.DataFrame(df_clean['issignal'], dtype='int')


#Creating a train and test set from the signal and background combined data set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=324)

dtrain = xgb.DMatrix(x_train, label = y_train)
dtest = xgb.DMatrix(x_whole, label = y_whole)
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
              'eta': 0.1,
              'eval_metric': 'auc', 'nthread' : 7}
    cv_result = xgb.cv(params, dtrain, num_boost_round=70, nfold=5)
    return  cv_result['test-auc-mean'].iloc[-1]


#Invoking the Bayesian Optimizer with the specified parameters to tune
xgb_bo = BayesianOptimization(bo_tune_xgb, {'max_depth': (4, 10),
                                             'gamma': (0, 1),
                                            'alpha': (2,20),
                                             'learning_rate':(0,1),
                                             'n_estimators':(100,500)
                                            })

#performing Bayesian optimization for 5 iterations with 8 steps of random exploration with an #acquisition function of expected improvement
xgb_bo.maximize(n_iter=5, init_points=8, acq='ei')

max_param = xgb_bo.max['params']
param= {'alpha': max_param['alpha'], 'gamma': max_param['gamma'], 'learning_rate': max_param['learning_rate'], 'max_depth': int(round(max_param['max_depth'],0)), 'n_estimators': int(round(max_param['n_estimators'],0)), 'objective': 'binary:logistic'}

bst = xgb.train(param, dtrain)
bst1= bst.predict(dtrain)

bst_test = pd.DataFrame(data=bst.predict(dtest1),  columns=["xgb_preds"])
y_test=y_test.set_index(np.arange(0,bst_test.shape[0]))
bst_test['issignal']=y_test['issignal']

#To visualize the predictions of the classifier in terms of probabilities
#The first argument should be a data frame, the second a column in it, in the form 'preds'
preds_prob(bst_test,'xgb_preds', 'issignal')

#To choose the best threshold
train_best, test_best = AMS(y_train, bst1,y_test, bst_test['xgb_preds'])

#Applying XGB on the 10k events data-set
df_clean['xgb_preds'] = bst.predict(dtest)

#plot confusion matrix
cut1 = test_best
df_clean['xgb_preds1'] = ((df_clean['xgb_preds']>cut1)*1)
cnf_matrix = confusion_matrix(y_whole, df_clean['xgb_preds1'], labels=[1,0])
#cnf_matrix = confusion_matrix(new_check_set['issignal'], new_check_set['new_signal'], labels=[1,0])
np.set_printoptions(precision=2)
fig, axs = plt.subplots(figsize=(10, 8))
axs.yaxis.set_label_coords(-0.04,.5)
axs.xaxis.set_label_coords(0.5,-.005)
plot_confusion_matrix(cnf_matrix, classes=['signal','background'], title='Confusion Matrix for XGB for cut > '+str(cut1))


#comparison with manual cuts
#returns a new df 
new_check_set=KFPF_lambda_cuts(df_clean)


cut3 = test_best
mask1 = df_clean['xgb_preds']>cut3
df3_base=df_clean[mask1]
from matplotlib import gridspec
range1= (1.0999, 1.17)
fig, axs = plt.subplots(2, 1,figsize=(15,10), sharex=True,  gridspec_kw={'width_ratios': [10],
                           'height_ratios': [8,4]})

ns, bins, patches=axs[0].hist((df3_base['mass']),bins = 300, range=range1, facecolor='red',alpha = 0.3)
ns1, bins1, patches1=axs[0].hist((new_check_set['mass']),bins = 300, range=range1,facecolor='blue',alpha = 0.3)

axs[0].set_ylabel("counts", fontsize = 15)
axs[0].legend(('XGBoost Selected $\Lambda$s','KFPF selected $\Lambda$s'), fontsize = 15, loc='upper right')

#plt.rcParams["legend.loc"] = 'upper right'
axs[0].set_title("The lambda's Invariant Mass histogram with KFPF and XGB selection criteria on KFPF variables", fontsize = 15)
axs[0].grid()
axs[0].tick_params(axis='both', which='major', labelsize=15)


hist1, bin_edges1 = np.histogram(df3_base['mass'],range=(1.09, 1.17), bins=300)
hist2, bin_edges2 = np.histogram(new_check_set['mass'],range=(1.09, 1.17), bins=300)

#makes sense to have only positive values 
diff = (hist1 - hist2)
axs[1].bar(bins[:-1],     # this is what makes it comparable
        ns / ns1, # maybe check for div-by-zero!
        width=0.001)
plt.xlabel("Mass in $\dfrac{GeV}{c^2}$", fontsize = 15)
axs[1].set_ylabel("XGB / KFPF", fontsize = 15)
axs[1].grid()
axs[1].tick_params(axis='both', which='major', labelsize=15)

plt.show()
fig.tight_layout()
