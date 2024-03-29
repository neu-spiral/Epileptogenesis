#%% import libraries
from models import *
import os
import random
from umap.umap_ import UMAP
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import pandas as pd
import numpy as np
import argparse

seed_value= 42
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

outer_splits=5
cv_outer=StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=seed_value)
processed_data_path="../_data/processed"
output_path=""
# Parse model, imputer, neighbors(KNN) values from user
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='all')
parser.add_argument('--features', type=str, default='bar')
args = parser.parse_args()

#%% Load data
X_df,y_df=load_data_df(processed_data_path)
y=y_df.to_numpy()
fmri_key=3

x_fmri=X_df.iloc[:,1+((fmri_key-1)*166):1+(fmri_key*166)]
x_dmri=X_df.iloc[:,499:562]
x_eeg=X_df.iloc[:,564:567]
merged_df = x_fmri.join(x_dmri)
X_name = merged_df.join(x_eeg)

# Find the columns that contain the substring '_A'
fmri_o = X_name.columns.str.contains('Overlap AAL3.')
fmri_o1 = X_name.columns.str.contains('cluster0')
fmri_o2 = X_name.columns.str.contains('cluster')
# fmri_p = X_name.columns.str.contains('Mean Str Pos AAL3.cluster')

# Rename the columns that contain the above substrings
X_name = X_name.rename(columns={old_col: old_col.replace('Overlap AAL3.', '') for old_col in X_name.columns[fmri_o]})
X_name = X_name.rename(columns={old_col: old_col.replace('cluster0', 'Cluster ') for old_col in X_name.columns[fmri_o1]})
X_name = X_name.rename(columns={old_col: old_col.replace('cluster', 'Cluster ') for old_col in X_name.columns[fmri_o2]})
# X_name = X_name.rename(columns={old_col: old_col.replace('Mean Str Pos AAL3.cluster', 'Cluster ') for old_col in X_name.columns[fmri_p]})


#%% Cross validate with Shap values
if args.model=='all':
    all_shap_values = []
    all_test_instances = []
    for k, (train_idx,test_idx) in tqdm(enumerate(cv_outer.split(X_df,y))):
        # if k>0:
        #     break
        y_train = y[train_idx]
        y_test = y[test_idx]
        X_train_raw = X_df.iloc[train_idx]
        X_test_raw = X_df.iloc[test_idx]
        
        # Imputing the train and test splits
        X_train,X_test=impute_data_df(imputer='KNN',train_df=X_train_raw,test_df=X_test_raw,fmri_key=fmri_key, neighbors=1)

        # Create a new DataFrame with the imputed values
        X_train = pd.DataFrame(X_train, columns=X_name.columns)
        X_test = pd.DataFrame(X_test, columns=X_name.columns)

        clf=None
        clf=make_pipeline(StandardScaler(), AdaBoostClassifier(n_estimators=50))
        # clf=make_pipeline(StandardScaler(), SVC(gamma='auto'))
        # clf=make_pipeline(StandardScaler(), SVC(gamma='auto',probability=True))
        
        clf.fit(X_train,y_train)
        # f = lambda x: clf.predict_proba(x)[:,1]
        f = lambda x: clf.predict(x)

        explainer = shap.Explainer(f,X_test)
        shap_values = explainer.shap_values(X_test)

        all_test_instances.append(X_test)
        all_shap_values.append(shap_values)

    # Normalized AdB
    # Concatenate all SHAP values and test instances
    all_shap_values = np.vstack(all_shap_values)
    all_test_instances = pd.concat(all_test_instances)

#%% Cross validate with SFS & Shap values  
if args.model=='SFS':
    all_shap_values = pd.DataFrame(columns=X_name.columns)
    all_test_instances = pd.DataFrame(columns=X_name.columns)
    all_umap=[]
    y_collect=[]
    rho=0.5

    for k, (train_idx,test_idx) in tqdm(enumerate(cv_outer.split(X_df,y))):
        # if k>0:
        #     break
        y_train = y[train_idx]
        y_test = y[test_idx]
        X_train_raw = X_df.iloc[train_idx]
        X_test_raw = X_df.iloc[test_idx]
        
        # Imputing the train and test splits
        X_train,X_test=impute_data_df(imputer='KNN',train_df=X_train_raw,test_df=X_test_raw,fmri_key=fmri_key, neighbors=1)

        # Create a new DataFrame with the imputed values
        X_train = pd.DataFrame(X_train, columns=X_name.columns)
        X_test = pd.DataFrame(X_test, columns=X_name.columns)

        clf=None
        clf=make_pipeline(StandardScaler(), SVC(gamma='auto'))

        sfs=None
        sfs=SequentialFeatureSelector(clf, direction='forward', scoring='roc_auc', n_features_to_select=5)
        sfs.fit(X_train,y_train)
        # Get the names of the selected features
        selected_features = X_train.columns[sfs.get_support()]
        print(selected_features)
        
        X_train=X_train[selected_features]
        X_test=X_test[selected_features]
        
        # check_test=X_test
        # check_rank=sfs.transform(X_test)

        # # # removing features above a certain mutual correlation coefficient
        # keep_col=[]
        # for j in range(X_train.shape[1]):
        #     keep_col.append(j)
        #     if j==0:
        #         continue
        #     else:
        #         for m in range(j):
        #             r_coef=pearsonr(X_train.iloc[:,m],X_train.iloc[:,j])[0]
        #             if r_coef>=rho:
        #                 keep_col.remove(j)
        #                 break

        # X_train=X_train.iloc[:,keep_col]
        # X_test=X_test.iloc[:,keep_col]
        # print(X_train.columns)

        clf.fit(X_train,y_train)
        f = lambda x: clf.predict(x)

        explainer = shap.KernelExplainer(f,X_test)
        # explainer = shap.Explainer(f,X_test)
        shap_values = explainer.shap_values(X_test)

        # Create a new DataFrame with the NumPy array data and selected column names
        new_df = pd.DataFrame(shap_values, columns=X_train.columns)

        all_test_instances=pd.concat([all_test_instances, X_test])
        # all_shap_values.append(shap_values)
        all_shap_values=pd.concat([all_shap_values, new_df])

        reducer = None
        reducer = UMAP()
        reducer.fit(X_train)
        embedding = reducer.transform(X_test)
        for i in range(embedding.shape[0]):
            all_umap.append(embedding[i])
            y_collect.append(y_test[i])

    # Normalized SFS & Shap values
    # Concatenate all SHAP values and test instances
    all_shap_values = all_shap_values.fillna(0)
    all_test_instances = all_test_instances.fillna(0)
    all_shap_values = all_shap_values.to_numpy()

#%% Plotting
# Number of features to display in the summary plot
num_features = 10

# Calculate mean absolute SHAP values and select top N features
mean_abs_shap_values = np.mean(np.abs(all_shap_values), axis=0)
top_feature_indices = np.argsort(mean_abs_shap_values)[-num_features:]

# Filter SHAP values and test instances based on the top N features
filtered_shap_values = all_shap_values[:, top_feature_indices]
filtered_test_instances = all_test_instances.iloc[:, top_feature_indices]

# if args.features=='bar':
# Create a summary plot with the top N features
shap.summary_plot(filtered_shap_values, filtered_test_instances, plot_type="bar")
# else:
# Create a swarm plot with the top N features
shap.summary_plot(filtered_shap_values, filtered_test_instances)
# plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')
# plt.show()

#%% UMAP
if args.model=='UMAP':
    reducer = UMAP()
    X_train=impute_data_df(imputer='KNN',train_df=X_df,fmri_key=fmri_key, neighbors=1)

    # Create a new DataFrame with the imputed values
    # X_train = pd.DataFrame(X_train, columns=X_name.columns)
    scaled_data = StandardScaler().fit_transform(X_train)
    embedding = reducer.fit_transform(scaled_data)

    # Create a DataFrame with the embedding and the labels
    embedding_df = pd.DataFrame(embedding, columns=['Component 1', 'Component 2'])
    embedding_df['Label'] = y

    # Create the seaborn scatter plot
    sns.scatterplot(data=embedding_df, x='Component 1', y='Component 2', hue='Label', palette=['red', 'blue'])

    # Display the plot
    plt.gca().set_aspect('equal', 'datalim')
    plt.show()

#%% UMAP-SFS
# Create a DataFrame with the embedding and the labels
# embedding_df = pd.DataFrame(all_umap, columns=['Component 1', 'Component 2'])
# embedding_df['Label'] = y_collect

# # Create the seaborn scatter plot
# sns.scatterplot(data=embedding_df, x='Component 1', y='Component 2', hue='Label', palette=['red', 'blue'])

# # Display the plot
# plt.gca().set_aspect('equal', 'datalim')
# plt.show()

# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=[sns.color_palette()[x] for x in y])
# plt.title('UMAP projection of the Penguin dataset', fontsize=24);


#%% Usage
# python shap_support.py --model all --features 10