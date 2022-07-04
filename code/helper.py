#%%
import numpy as np
# %matplotlib inline
import matplotlib.pyplot as plt 
from matplotlib.pyplot import imshow
from matplotlib.lines import Line2D
import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn import decomposition, linear_model,metrics
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder,FunctionTransformer
class_labels = LabelEncoder()
from sklearn.model_selection import cross_val_score,GridSearchCV,StratifiedKFold,KFold,train_test_split,LeaveOneOut
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import classification_report, confusion_matrix,mean_squared_error,r2_score
from sklearn.metrics import auc, RocCurveDisplay, roc_curve, f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import RFE, SelectKBest, f_classif, chi2, mutual_info_classif
from sklearn.manifold import TSNE
from sklearn.naive_bayes import GaussianNB
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer,KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import CCA
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin

from astropy.stats import jackknife_resampling, jackknife_stats, binom_conf_interval
from tqdm import tqdm
import math
from itertools import product
from contextlib import redirect_stdout
import pandas as pd
import time
import scipy
from scipy import io, stats
from statistics import mean
#from astropy.stats import jackknife_resampling, jackknife_stats, binom_conf_interval
from extra.MMIDimReduction import MMINet
from cluster.selfrepresentation import ElasticNetSubspaceClustering, SparseSubspaceClusteringOMP
from cca_zoo.models import GCCA, KGCCA
# from extra.gcca import GCCA

seed_value= 42
import os
os.environ['PYTHONHASHSEED']=str(seed_value)
import random
random.seed(seed_value)
np.random.seed(seed_value)


#%%
def remove_non_features(column_list):
    '''Removes column names that aren't features from column list'''
    for to_remove in ["ID","Late Seizure Label","Subject","Subject Number"]:
        if to_remove in column_list:
            column_list.remove(to_remove)
    return column_list

def modality_svm(selecter,imputer,x_df,y):
    ''' Fits and scores a kPCA+SVM classifier for each modality, based on an imputer'''

    svm_classifier = Pipeline([("pca",KernelPCA()), ("svm",SVC(probability=True))])
    param_grid_svm={"clf__pca__n_components":[2,3,4,5,None],"clf__pca__gamma":[.01,.05,.1],"clf__pca__kernel":["linear","rbf"],
    "clf__svm__C": [1, 10, 100], "clf__svm__gamma": [.01, .1]}
    
    pipe=Pipeline([("select",selecter),("scale",StandardScaler()), ("impute",imputer),("clf",svm_classifier)])

    search=GridSearchCV(estimator=pipe,scoring=score_string,param_grid=param_grid_svm,cv=cv_inner,refit=True).fit(x_df,y)

    scores= cross_val_score(search, x_df, y, scoring=score_string, cv=cv_outer, n_jobs=-1)

    return search,scores

def modality_tree(selecter,imputer,x_df,y):
    ''' Fits and scores a tree-based classifier for each modality, based on an imputer'''

    tree_classifier= Pipeline([("kbest",SelectKBest(chi2)), ("tree",AdaBoostClassifier())])
    param_grid_tree={"clf__kbest__k":[2,3,5,7,"all"],"clf__tree__n_estimators":[10,50,100]}
    
    pipe=Pipeline([("select",selecter), ("impute",imputer),("clf",tree_classifier)])

    search=GridSearchCV(estimator=pipe,scoring=score_string,param_grid=param_grid_tree,cv=cv_inner,refit=True).fit(x_df,y)

    scores= cross_val_score(search, x_df, y, scoring=score_string, cv=cv_outer, n_jobs=-1)

    return search,scores

def drop_nan_index(X,y,idx):
    ''' Selects a subset based on idx and then returns the subset of X,y that correspond to rows without any nans. 
    For use in test train split for individual modality classifiers'''

    X_sub=X[idx,:]
    y_sub=y[idx]
    drop_rows=~np.isnan(X_sub).any(axis=1)
    X_sub=X_sub[drop_rows]
    y_sub=y_sub[drop_rows]

    return X_sub,y_sub

def nb_svm(x,y):
    ''' Fits and scores a kPCA+SVM classifier for use in a naive bayes classifier'''

    svm_classifier = Pipeline([("pca",KernelPCA()), ("svm",SVC(probability=True))])
    param_grid_svm={"clf__pca__n_components":[2,3,5,None],"clf__pca__gamma":[.01,.1],"clf__pca__kernel":["linear","rbf"]}

    # "clf__svm__C": [1, 10, 100], "clf__svm__gamma": [.01, .1]
    
    pipe=Pipeline([("scale",StandardScaler()),("clf",svm_classifier)])

    search=GridSearchCV(estimator=pipe,scoring="f1",param_grid=param_grid_svm,cv=5,refit=True).fit(x,y)

    return search

def nb_tree(x,y):
    ''' Fits and scores a tree-based classifier for use in a naive bayes classifier'''

    tree_classifier= Pipeline([("kbest",SelectKBest(chi2)), ("tree",AdaBoostClassifier())])
    param_grid_tree={"clf__kbest__k":[2,3,5,7,10,15],"clf__tree__n_estimators":[10,50,100]}
    
    pipe=Pipeline([("scale",StandardScaler()),("clf",tree_classifier)])

    search=GridSearchCV(estimator=pipe,scoring=score_string,param_grid=param_grid_tree,cv=cv_inner,refit=True).fit(x,y)

    # scores= cross_val_score(search, x_df, y, scoring=score_string, cv=cv_outer, n_jobs=-1)

    return search

def naive_bayes_multimodal(fmri_class,X_fmri,dwi_class,X_dwi,y_test,y_train,eeg_class=np.nan,X_eeg=np.nan):
    '''Makes a prediction based on a naive bayes multimodal fusion using a conditional independence assumption, which ignores modalities that don't have features for a given subject'''
    p_true=sum(y_test)/len(y_test)
    # p_true=sum(y_train)/len(y_train)
    p_false=1-p_true

    n_subs=X_fmri.shape[0]
    #The following two variable will not actually be probabilities (they shouldn't sum to 1). Essentially this function uses the approximation 
    # p(x|l) \approx p(l|x)/p(l). To get a real generative model, I'd suggest just using a MLE for a gaussian model for the fMRI and dwi, and a poisson model for EEG
    y_prob_false=[]
    y_prob_true=[]
    predict=[]

    for row in range(n_subs):
        if np.isnan(X_fmri[row,:]).any(): #check if there's fMRI data, if not set the relative prob to 1
            fmri_prob_true=1
            fmri_prob_false=1
        else: 
            fmri_prob_false=fmri_class.predict_proba(X_fmri[row,:].reshape(1, -1))[0][0]/p_false 
            fmri_prob_true=fmri_class.predict_proba(X_fmri[row,:].reshape(1, -1))[0][1]/p_true

        if np.isnan(X_dwi[row,:]).any():
            dwi_prob_true=1
            dwi_prob_false=1
        else: 
            dwi_prob_false=dwi_class.predict_proba(X_dwi[row,:].reshape(1, -1))[0][0]/p_false
            dwi_prob_true=dwi_class.predict_proba(X_dwi[row,:].reshape(1, -1))[0][1]/p_true

        # if np.isnan(X_eeg):
        #     eeg_prob_true=1
        #     eeg_prob_false=1            
        if np.isnan(X_eeg[row,:]).any():
            eeg_prob_true=1
            eeg_prob_false=1
        else: 
            eeg_prob_false=eeg_class.predict_proba(X_eeg[row,:].reshape(1, -1))[0][0]/p_false
            eeg_prob_true=eeg_class.predict_proba(X_eeg[row,:].reshape(1, -1))[0][1]/p_true
        
        prob_false=fmri_prob_false*dwi_prob_false*eeg_prob_false*p_false
        y_prob_false.append(prob_false)
        prob_true=fmri_prob_true*dwi_prob_true*eeg_prob_true*p_true
        y_prob_true.append(prob_true)
        predict.append(prob_true>=prob_false) #check which "probability" is higher. Could test whether taking the tie break the other direction (i.e. setting the prediciton to prob_true>=prob_false) changes the results

    return predict,y_prob_true,y_prob_false
    
def nb_cca_svm(x,y):
    ''' Fits and scores a CCA+SVM classifier for use in a Bayes fusion classifier'''

    svm_classifier = Pipeline([("svm",SVC(probability=True))])
    param_grid_svm={"clf__svm__gamma": ['auto']}
    # param_grid_svm={"clf__svm__C": [10], "clf__svm__gamma": [.01]}
    
    pipe=Pipeline([("scale",StandardScaler()),("clf",svm_classifier)])

    search=GridSearchCV(estimator=pipe,scoring=score_string,param_grid=param_grid_svm,cv=cv_inner,refit=True).fit(x,y)

    return search

def load_data(processed_data_path,NBF=False):
    fmri_features=pd.read_csv(f"{processed_data_path}/fMRI/fMRI_features_AAL.csv",index_col=0)

    # print('fMRI Subject IDs')
    # print(fmri_features["Subject"])
    dwi_features = pd.read_csv(f"{processed_data_path}/DWI/subs_jan_2022.csv")

    dwi_features["Subject"]=dwi_features["ID"].str[:9]
    dwi_features["Late Seizure Label"]=dwi_features["Label"]
    dwi_features=dwi_features.drop("Label",axis=1)

    eeg_features=pd.read_csv(f"{processed_data_path}/EEG/EEG_features_v0.csv",index_col=0)

    # need to load EEG and DWI features and sort out which subjects to use programatically

    all_features_df=fmri_features.set_index("Subject").join(dwi_features.set_index("Subject"),how="outer",lsuffix=" fMRI",rsuffix=" DWI").reset_index()
    all_features_df=all_features_df.set_index("Subject").join(eeg_features.set_index("Subject"),how="outer",lsuffix=" Mix",rsuffix=" EEG").reset_index()
    all_features_df["Late Seizure Label EEG"]=all_features_df["Late Seizure Label"]


    all_features_df["Late Seizure Label"]=(all_features_df["Late Seizure Label fMRI"].fillna(0)+all_features_df["Late Seizure Label DWI"].fillna(0)+all_features_df["Late Seizure Label EEG"].fillna(0))>0

    # make np array features for classification

    dwi_columns=remove_non_features([*dwi_features])
    eeg_columns=remove_non_features([*eeg_features])
    fmri_columns=remove_non_features([*fmri_features])

    #dwi 
    y = all_features_df["Late Seizure Label"].to_numpy()
    X_dwi = all_features_df[dwi_columns].to_numpy()

    # fMRI

    overlap_columns=[]
    mean_str_pos_columns=[]
    mean_str_neg_columns=[]

    for col in fmri_columns:
        if "Overlap AAL" in col:
            overlap_columns.append(col)
        elif "Pos AAL" in col:
            mean_str_pos_columns.append(col)
        elif "Neg AAL" in col:
            mean_str_neg_columns.append(col)

    X_over_aal=all_features_df[overlap_columns].to_numpy()
    X_pos_str_aal=all_features_df[mean_str_pos_columns].to_numpy()

    all_features_df[mean_str_neg_columns]=-1*all_features_df[mean_str_neg_columns]
    X_neg_str_aal=all_features_df[mean_str_neg_columns].to_numpy()
    #eeg 
    X_eeg=all_features_df[eeg_columns].to_numpy()

    #all_features=np.concatenate([X_over_aal,X_dwi,X_eeg],axis=1)
    fmri_len=X_over_aal.shape[1]
    dwi_len=X_dwi.shape[1]
    eeg_len=X_eeg.shape[1]

    fmri_ind=[*range(0,fmri_len)]
    dwi_ind=[*range(fmri_len,fmri_len+dwi_len)]
    eeg_ind=[*range(fmri_len+dwi_len,fmri_len+dwi_len+eeg_len)]

    select_fmri_ov=ColumnTransformer([("fMRI ov",'passthrough',overlap_columns)])
    select_fmri_pos=ColumnTransformer([("fMRI pos",'passthrough',mean_str_pos_columns)])
    select_fmri_neg=ColumnTransformer([("fMRI neg",'passthrough',mean_str_neg_columns)])
    select_fmri=ColumnTransformer([("fMRI",'passthrough',[*mean_str_pos_columns,*mean_str_neg_columns,*overlap_columns])])

    select_eeg=ColumnTransformer([("EEG",'passthrough',eeg_columns)])
    select_dwi=ColumnTransformer([("DWI",'passthrough',dwi_columns)])

    select_all_pos=ColumnTransformer([("ALL",'passthrough',[*mean_str_pos_columns,*eeg_columns,*dwi_columns])])
    select_all_neg=ColumnTransformer([("ALL",'passthrough',[*mean_str_neg_columns,*eeg_columns,*dwi_columns])])
    select_all_ov=ColumnTransformer([("ALL",'passthrough',[*overlap_columns,*eeg_columns,*dwi_columns])])

    # Fusion excluding EEG
    dwi_fmri_pos=ColumnTransformer([("ALL",'passthrough',[*mean_str_pos_columns,*dwi_columns])])
    dwi_fmri_neg=ColumnTransformer([("ALL",'passthrough',[*mean_str_neg_columns,*dwi_columns])])
    dwi_fmri_ov=ColumnTransformer([("ALL",'passthrough',[*overlap_columns,*dwi_columns])])

    select_all=ColumnTransformer([("ALL",'passthrough',[*mean_str_pos_columns,*mean_str_neg_columns,*overlap_columns,*eeg_columns,*dwi_columns])])
    X_df=all_features_df
    X_df=X_df.drop(["ID","Subject","Subject Number"],axis=1)
    # X_df=X_df[:].values
    X_df=X_df.to_numpy()

    if NBF:
        return X_dwi,X_eeg,X_neg_str_aal,X_pos_str_aal,X_over_aal,y
    else:
        return X_df,y

def impute_data(imputer,train_df,fmri_key,neighbors=3):
    iter_estimator=RandomForestRegressor(
        n_estimators=4,
        max_depth=10,
        bootstrap=True,
        max_samples=0.5,
        n_jobs=2,
        random_state=seed_value)
    if imputer=="KNN":
        imputer_mode=KNNImputer(n_neighbors=neighbors)
    elif imputer=="Iterative":
        imputer_mode=IterativeImputer(random_state=seed_value,estimator=iter_estimator,sample_posterior=False,max_iter=5)

    # Select appropriate columns for chosen fmri strength
    # fmri_label=col_0, dmri_label=col_562, common_label=col_563, eeg_label=col_567

    x_fmri=train_df[:,1+((fmri_key-1)*166):1+(fmri_key*166)]
    x_dmri=train_df[:,499:562]
    x_eeg=train_df[:,564:567]
    train_df_f=np.append(x_fmri,x_dmri,axis=1)
    train_df_f=np.append(train_df_f,x_eeg,axis=1)

    # x_fmri=test_df[:,1+((fmri_key-1)*166):1+(fmri_key*166)]
    # x_dmri=test_df[:,499:562]
    # x_eeg=test_df[:,564:567]
    # test_df_f=np.append(x_fmri,x_dmri,axis=1)
    # test_df_f=np.append(test_df_f,x_eeg,axis=1)
    
    imputed_train=imputer_mode.fit_transform(train_df_f)
    # imputed_test=imputer_mode.transform(test_df_f)

    # return imputed_train,imputed_test
    return imputed_train
