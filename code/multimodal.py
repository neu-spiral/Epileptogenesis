#%%
from loader import *

# Load data
processed_data_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/_data/processed"
output_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/code"

class_features_df,y=load_data(processed_data_path)
class_features_df=class_features_df.to_numpy()

# Setup parameters    
outer_splits=5
inner_splits=5
score_string="f1"

cv_outer=StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed_value)
cv_inner=StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=seed_value)

#set up imputers for missing data
# imputers=[KNNImputer(n_neighbors=3),IterativeImputer(random_state=42)]
# imputer_strs=["KNN","Iterative"]
# imputed_df=impute_data("KNN",class_features_df)

#%%
fmri_feat=['Neg','Pos','Ov']
nb_seq_cca_results=pd.DataFrame()
j = 0
imputer="KNN"
neighbors=3
nb_seq_results=pd.DataFrame()

for key in fmri_feat:
    j+=1
    print(j,'-th fmri strength running')

    # Varying no. of greedy features
    for i in tqdm(range(10)):
        f1_scores=[]
        direction='forward'
        # direction='backward'

        for train_idx, test_idx in cv_outer.split(class_features_df,y):

            X_train = class_features_df[train_idx,:]
            X_test = class_features_df[test_idx,:]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            # Imputing the train and test splits
            X_train_imputed,X_test_imputed=impute_data(imputer,X_train,X_test,fmri_key=j, neighbors=neighbors)

            # SVM Classifier
            clf=None
            # clf = make_pipeline(StandardScaler(), SVC(C=10,gamma=0.01))
            clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

            # Sequential feature selector
            sfs=None
            sfs = SequentialFeatureSelector(clf, direction=direction, scoring='f1', n_features_to_select=i+1)

            sfs.fit(X_train_imputed, y_train)
            X_train_seq = sfs.transform(X_train_imputed)
            X_test_seq = sfs.transform(X_test_imputed)

            clf.fit(X_train_seq, y_train)
            y_pred = clf.predict(X_test_seq)

            f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

        nb_seq_results = nb_seq_results.append({'fMRI':key,'Features':i+1,'Direction':direction,'Imputer':imputer,'Neighbors':neighbors,'Mean f1':mean(f1_scores),'SEM f1':1.96*stats.sem(f1_scores,ddof=0)},ignore_index=True)
        
nb_seq_results.to_csv(f"{output_path}/_stats/sfs.csv")
# %%
