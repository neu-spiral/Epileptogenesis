#%% import libraries
from models import *

#%% Setup parameters    
outer_splits=5
inner_splits=5
cv_outer=StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed_value)
cv_inner=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed_value)
score_string="roc_auc"
# earlier choice was 'f1'
processed_data_path="/home/navid/Dropbox/Repo_2023/Epilep/Epileptogenesis/_data/processed"
# processed_data_path='/Users/Navid1/Dropbox/Repo_2023/Epilep/Epileptogenesis/_data/processed' # mac
output_path="/home/navid/Dropbox/Repo_2023/Epilep/Epileptogenesis/code/"
# output_path='/Users/Navid1/Dropbox/Repo_2023/Epilep/Epileptogenesis/code/' # mac

#%% Load data
X_df,y=load_data(processed_data_path)
# X_train=impute_data(imputer='KNN',train_df=X_df)
    # x_fmri=train_df.iloc[:,1+((fmri_key-1)*166):1+(fmri_key*166)]
    # x_dmri=train_df.iloc[:,499:562]
    # x_eeg=train_df.iloc[:,564:567]
    # merged_df = x_fmri.join(x_dmri)
    # train_df_f = merged_df.join(x_eeg)

for k, (train_idx,test_idx) in tqdm(enumerate(cv_outer.split(X_df,y))):
    
    y_train = y[train_idx]
    y_test = y[test_idx]
    X_train_raw = X_df[train_idx]
    X_test_raw = X_df[test_idx]
    
    # Imputing the train and test splits
    X_train,X_test=impute_data(imputer='KNN',train_df=X_train_raw,test_df=X_test_raw,fmri_key=3, neighbors=1)
    # clf=AdaBoostClassifier(n_estimators=50)
    clf=SVC(gamma='auto')
    print('y: ',np.count_nonzero(np.isnan(y)))
    print('X_train: ',np.count_nonzero(np.isnan(X_train)))
    clf.fit(X_train,y_train)

    explainer = shap.KernelExplainer(model =clf.predict,data = X_train)
    shap_values = explainer.shap_values(X_test)
    shap.plots.beeswarm(shap_values)

# X_df=X_df.to_numpy()


#%% Parse model, imputer, neighbors(KNN) values from user
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='SFS')
parser.add_argument('--imputer', type=str, default='KNN')
parser.add_argument('--neighbors', type=int, default=1)
parser.add_argument('--text', type=str, default='_')
parser.add_argument('--roc_flag', type=str, default='False')
parser.add_argument('--fixed_feat', type=int, default=0)
parser.add_argument('--rho', type=float, default=1.0)
parser.add_argument('--options', type=str, default='None')
args = parser.parse_args()

# model_list=['SFS','CCA','NBF']
model=args.model
# imputer_list=["KNN","Iterative"]
imputer=args.imputer
# neighbors_list=[1,2,3,5]
neighbors=args.neighbors
# text_list=['_sample_posterior=True,max_iter=5']
text=args.text
roc_flag=args.roc_flag
fixed_feat=args.fixed_feat
# options_list=['roc_data','y_info','X_test']
options=args.options
rho=args.rho

#%%
# run_estimator(cv_outer,output_path,model,X_df,y,text,options,imputer,neighbors,roc_flag,fixed_feat,rho)
# plot_manifold(output_path,model,X_df,y,text,manifold_opts,imputer,neighbors,fixed_feat)

# _,_=load_data(processed_data_path)

# %%
# if manifold_flag:
#     plot_manifold(output_path,model,X_df,y,text,imputer,neighbors,fixed_feat)

# python main.py --model SFS --roc_flag True --rho 0.7 --text _svm_scale

# python main.py --model NBF --text _adb_fs

# python main.py --model CCA+SFS --roc_flag True --fixed_feat 7 --options roc_data --rho 0.5 --text _reproduce
# conda create --name epilep --file requirements.txt 