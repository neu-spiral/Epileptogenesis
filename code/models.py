from helper import *

def run_estimator(cv_outer,output_path,model,class_features_df,y,text,
                    imputer='KNN',neighbors=3,direction='forward'):
    # fmri_feat=['Neg','Pos','Ov']
    fmri_feat=['Ov']
    j=2
    results=pd.DataFrame()
    print('Model:',model,', Imputer:',imputer,', Neighbors:',neighbors)

    fig, ax = plt.subplots()
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for key in fmri_feat:
        j+=1
        print(key,'fmri strength running')

        # Varying no. of greedy features
        # for i in tqdm(range(10)):
        for i in tqdm([3]):
            f1_scores=[]

            for k, (train_idx,test_idx) in tqdm(enumerate(cv_outer.split(class_features_df,y))):

                X_train = class_features_df[train_idx,:]
                X_test = class_features_df[test_idx,:]
                y_train = y[train_idx]
                y_test = y[test_idx]
                
                # Imputing the train and test splits
                X_train_imputed,X_test_imputed=impute_data(imputer,X_train,X_test,fmri_key=j, neighbors=neighbors)

                # Fit and test with desired classifier
                y_pred,fpr,tpr,roc_auc=model_run(model,i,k,ax,X_train_imputed,X_test_imputed,y_train,y_test,direction='forward')
                interp_tpr = np.interp(mean_fpr,fpr,tpr)
                interp_tpr[0] = 0.0
                tprs.append(interp_tpr)
                aucs.append(roc_auc)
                f1_scores.append(f1_score(y_test, y_pred, average='weighted'))

            results = results.append({'fMRI':key,'Features':i+1,'Direction':direction,'Imputer':imputer,'Neighbors':neighbors,'Mean f1':mean(f1_scores),'SEM f1':1.96*stats.sem(f1_scores,ddof=0)},ignore_index=True)

            ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            ax.plot(
                mean_fpr,
                mean_tpr,
                color="b",
                label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
                lw=2,
                alpha=0.8,
            )
            # std_tpr = np.std(tprs, axis=0)
            # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

            # Using Binomial conf intervals, as laid out in Sourati 2015
            [tprs_upper, tprs_lower] = binom_conf_interval(mean_tpr*48, 48, confidence_level=0.95, interval='wilson')  

            ax.fill_between(
                mean_fpr,
                tprs_lower,
                tprs_upper,
                color="grey",
                alpha=0.2,
                # label=r"$\pm$ 1 std. dev.",
                label=r'95% level of confidence',
            )

            ax.set(
                xlim=[-0.05, 1.05],
                ylim=[-0.05, 1.05],
                # title="Receiver operating characteristic example",
            )
            ax.legend(loc="lower right")
            plt.savefig(output_path+'_plot/'+model+'_'+imputer+'_'+str(neighbors)+text+'.png') 
            
    results.to_csv(output_path+'_stat/'+model+'_'+imputer+'_'+str(neighbors)+text+'.csv')

def model_run(model,i,k,ax,X_train_imputed,X_test_imputed,y_train,y_test,direction='forward'):
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
        smig.learn(X_train_imputed, y_train, num_epochs=20)
        X_train = smig.reduce(X_train_imputed)
        X_test = smig.reduce(X_test_imputed)

    elif model=='CCA':
        cca=None
        cca=CCA(n_components=i+1)
        X_train_f,X_train_d=cca.fit_transform(X_train_imputed[:,0:166],X_train_imputed[:,166:229])
        X_train=np.concatenate((X_train_f,X_train_d), axis=1)
        X_test_f,X_test_d=cca.transform(X_test_imputed[:,0:166],X_test_imputed[:,166:229])
        X_test=np.concatenate((X_test_f,X_test_d), axis=1)

    elif model=='CCA+SFS':
        cca=None
        cca=CCA(n_components=i+1)
        X_train_f,X_train_d=cca.fit_transform(X_train_imputed[:,0:166],X_train_imputed[:,166:229])
        X_train_cca=np.concatenate((X_train_f,X_train_d), axis=1)
        X_test_f,X_test_d=cca.transform(X_test_imputed[:,0:166],X_test_imputed[:,166:229])
        X_test_cca=np.concatenate((X_test_f,X_test_d), axis=1)
        sfs=None
        sfs=SequentialFeatureSelector(clf, direction=direction, scoring='f1', n_features_to_select=10)
        X_train_sfs=sfs.fit_transform(X_train_imputed,y_train)
        X_test_sfs=sfs.transform(X_test_imputed)
        X_train=np.concatenate((X_train_cca,X_train_sfs), axis=1)
        X_test=np.concatenate((X_test_cca,X_test_sfs), axis=1)

    elif model=='CCA+SMIG':
        cca=None
        cca=CCA(n_components=i+1)
        X_train_f,X_train_d=cca.fit_transform(X_train_imputed[:,0:166],X_train_imputed[:,166:229])
        X_train_cca=np.concatenate((X_train_f,X_train_d), axis=1)
        X_test_f,X_test_d=cca.transform(X_test_imputed[:,0:166],X_test_imputed[:,166:229])
        X_test_cca=np.concatenate((X_test_f,X_test_d), axis=1)
        smig=None
        smig=MMINet(input_dim=232, output_dim=4, net='linear')
        smig.learn(X_train_imputed, y_train, num_epochs=10)
        X_train_smig = smig.reduce(X_train_imputed)
        X_test_smig = smig.reduce(X_test_imputed)
        X_train=np.concatenate((X_train_cca,X_train_smig), axis=1)
        X_test=np.concatenate((X_test_cca,X_test_smig), axis=1)

    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)

    # ROC-AUC plot
    viz = RocCurveDisplay.from_estimator(
        clf,
        X_test,
        y_test,
        name="ROC fold {}".format(k+1),
        alpha=0.3,
        lw=1,
        ax=ax,
    )


    return y_pred,viz.fpr,viz.tpr,viz.roc_auc