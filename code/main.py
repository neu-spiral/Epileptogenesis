#%%
from models import *

# Load data
processed_data_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/_data/processed"
output_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/code/"

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
parser.add_argument('--imputer', type=str, required=True)
parser.add_argument('--neighbors', type=int, required=True)
parser.add_argument('--text', type=str)
parser.add_argument('--roc_flag', type=str)
parser.add_argument('--fixed_feat', type=int)
parser.add_argument('--options', type=str)
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

cv_outer=StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed_value)
cv_inner=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed_value)

run_estimator(cv_outer,output_path,model,X_df,y,text,options,imputer,neighbors,roc_flag,fixed_feat)
# plot_manifold(output_path,model,X_df,y,text,manifold_opts,imputer,neighbors,fixed_feat)

# %%
#set up imputers for missing data
# imputers=[KNNImputer(n_neighbors=3),IterativeImputer(random_state=42)]

        # if manifold_flag:
        #     plot_manifold(output_path,model,X_df,y,text,imputer,neighbors,fixed_feat)

# python main.py --model SFS --imputer KNN --neighbors 1 --roc_flag True --fixed_feat 0 --options y_info --text _y_info

# python plot_csv.py --name1 SFS_Ov_KNN_1_fixed_0_proba_y_info --name2 SFS_Ov_KNN_1_fixed_0_recc_proba_y_info

# python main.py --model NBF --imputer KNN --neighbors 1 --roc_flag False --fixed_feat 0 --options None --text _

# python main.py --model CCA+SFS --imputer KNN --neighbors 1 --roc_flag True --fixed_feat 7 --options roc_data --text _svm_sig_feat