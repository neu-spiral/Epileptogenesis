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
text='5_neigh'

cv_outer=StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed_value)
cv_inner=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed_value)

model='SFS'
run_estimator(cv_outer,output_path,model,class_features_df,y,text,neighbors=5)

# %%

#set up imputers for missing data
# imputers=[KNNImputer(n_neighbors=3),IterativeImputer(random_state=42)]
# imputer_strs=["KNN","Iterative"]
# imputed_df=impute_data("KNN",class_features_df)