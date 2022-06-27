from helper import *

def run_estimator(cv_outer,output_path,model,class_features_df,y,text,
                    imputer="KNN",neighbors=3,direction='forward'):
    fmri_feat=['Neg','Pos','Ov']
    # fmri_feat=['Neg']
    j = 0
    results=pd.DataFrame()
    for key in fmri_feat:
        j+=1
        print(key,'fmri strength running')

        # Varying no. of greedy features
        for i in tqdm(range(10)):
        # for i in tqdm([7]):
            f1_scores=[]

            for train_idx, test_idx in tqdm(cv_outer.split(class_features_df,y)):

                X_train = class_features_df[train_idx,:]
                X_test = class_features_df[test_idx,:]
                y_train = y[train_idx]
                y_test = y[test_idx]
                
                # Imputing the train and test splits
                X_train_imputed,X_test_imputed=impute_data(imputer,X_train,X_test,fmri_key=j, neighbors=neighbors)

                # Fit and test with desired classifier
                y_pred=model_run(model,i,X_train_imputed,X_test_imputed,y_train,direction='forward')
                f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

            results = results.append({'fMRI':key,'Features':i+1,'Direction':direction,'Imputer':imputer,'Neighbors':neighbors,'Mean f1':mean(f1_scores),'SEM f1':1.96*stats.sem(f1_scores,ddof=0)},ignore_index=True)
            
    results.to_csv(output_path+model+text+".csv")

def model_run(model,i,X_train_imputed,X_test_imputed,y_train,direction='forward'):
    # SVM Classifier
    clf=None
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

    if model=='SFS':
        sfs=None
        sfs = SequentialFeatureSelector(clf, direction=direction, scoring='f1', n_features_to_select=i+1)
        sfs.fit(X_train_imputed,y_train)
        X_train_seq = sfs.transform(X_train_imputed)
        X_test_seq = sfs.transform(X_test_imputed)

    clf.fit(X_train_seq, y_train)
    y_pred = clf.predict(X_test_seq)
    return y_pred