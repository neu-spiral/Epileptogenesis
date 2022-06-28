from helper import *

def run_estimator(cv_outer,output_path,model,class_features_df,y,text,
                    imputer='KNN',neighbors=3,direction='forward'):
    fmri_feat=['Neg','Pos','Ov']
    # fmri_feat=['Neg']
    j = 0
    results=pd.DataFrame()
    print('Model:',model,', Imputer:',imputer,', Neighbors:',neighbors)

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
            
    results.to_csv(output_path+model+'_'+imputer+'_'+str(neighbors)+text+".csv")

def model_run(model,i,X_train_imputed,X_test_imputed,y_train,direction='forward'):
    # SVM Classifier
    clf=None
    clf=make_pipeline(StandardScaler(), SVC(gamma='auto'))

    if model=='SFS':
        sfs=None
        sfs=SequentialFeatureSelector(clf, direction=direction, scoring='f1', n_features_to_select=i+1)
        X_train=sfs.fit_transform(X_train_imputed,y_train)
        X_test=sfs.transform(X_test_imputed)

    elif model=='SMIG':
        smig=None
        smig=MMINet(input_dim=232, output_dim=i+1, net='linear')
        smig.learn(X_train_imputed, y_train, num_epochs=10)
        X_train = smig.reduce(X_train_imputed)
        X_test = smig.reduce(X_test_imputed)

    elif model=='CCA':
        cca=None
        cca=CCA(n_components=i+1)
        X_train_f,X_train_d=cca.fit_transform(X_train_imputed[:,0:166],X_train_imputed[:,166:229])
        X_train=np.concatenate((X_train_f,X_train_d), axis=1)
        X_test_f,X_test_d=cca.transform(X_test_imputed[:,0:166],X_test_imputed[:,166:229])
        X_test=np.concatenate((X_test_f,X_test_d), axis=1)

    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    return y_pred