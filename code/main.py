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
score_string="f1"

# Parse model, imputer, neighbors(KNN) values from user
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--imputer', type=str, required=True)
parser.add_argument('--neighbors', type=int, required=True)
parser.add_argument('--text', type=str, required=True)
parser.add_argument('--roc', type=str)
parser.add_argument('--fixed_feat', type=int)
args = parser.parse_args()

# model_list=['SFS','CCA','NBF']
model=args.model
# imputer_list=["KNN","Iterative"]
imputer=args.imputer
# neighbors_list=[1,2,3,5]
neighbors=args.neighbors
# text_list=['_sample_posterior=True,max_iter=5']
text=args.text
roc=args.roc
fixed_feat=args.fixed_feat

cv_outer=StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed_value)
cv_inner=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed_value)

run_estimator(cv_outer,output_path,model,X_df,y,text,imputer,neighbors,roc,fixed_feat)

# %%
#set up imputers for missing data
# imputers=[KNNImputer(n_neighbors=3),IterativeImputer(random_state=42)]