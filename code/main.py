#%%
from models import *

# Load data
# processed_data_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/_data/processed"
processed_data_path='/Users/Navid1/Dropbox/Repo_2022/Epilep/Epileptogenesis/_data/processed' # mac
# output_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/code/"
output_path='/Users/Navid1/Dropbox/Repo_2022/Epilep/Epileptogenesis/code/' # mac

X_df,y=load_data(processed_data_path)
# X_df=X_df.to_numpy()

# Setup parameters    
outer_splits=5
inner_splits=5
score_string="roc_auc"
# earlier choice was 'f1'

# Parse model, imputer, neighbors(KNN) values from user
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
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

cv_outer=StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed_value)
cv_inner=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed_value)

run_estimator(cv_outer,output_path,model,X_df,y,text,options,imputer,neighbors,roc_flag,fixed_feat,rho)
# plot_manifold(output_path,model,X_df,y,text,manifold_opts,imputer,neighbors,fixed_feat)

# _,_=load_data(processed_data_path)

# %%
# if manifold_flag:
#     plot_manifold(output_path,model,X_df,y,text,imputer,neighbors,fixed_feat)

# python main.py --model SFS --roc_flag True --rho 0.7 --text _svm_scale

# python main.py --model NBF --text _adb_fs

# python main.py --model CCA+SFS --roc_flag True --fixed_feat 7 --options roc_data --rho 0.7 --text _f_d_svm_feats
