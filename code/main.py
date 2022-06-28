#%%
from models import *

# Load data
processed_data_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/_data/processed"
output_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/code/_stat/"

class_features_df,y=load_data(processed_data_path)
class_features_df=class_features_df.to_numpy()

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
args = parser.parse_args()

# model_list=['SFS','CCA']
model=args.model
# imputer_list=["KNN","Iterative"]
imputer=args.imputer
# neighbors_list=[1,2,3,5]
neighbors=args.neighbors
# text_list=['_sample_posterior=True,max_iter=5']
text=args.text

cv_outer=StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed_value)
cv_inner=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed_value)

run_estimator(cv_outer,output_path,model,class_features_df,y,text,imputer,neighbors=neighbors)

# %%

#set up imputers for missing data
# imputers=[KNNImputer(n_neighbors=3),IterativeImputer(random_state=42)]