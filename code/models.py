from helper import *

def run_estimator(cv_outer,output_path,model,X_df,y,text,
                    imputer='KNN',neighbors=3,roc_flag=False,fixed_feat=0,direction='forward'):
    fmri_feat=['Neg','Pos','Ov']
    # fmri_feat=['Ov']
    j=0
    results=pd.DataFrame()
    print('Model:',model,', Imputer:',imputer,', Neighbors:',neighbors)

    for key in fmri_feat:
        j+=1
        # if j>2:
            # break
        print(key,'fmri strength running')
        X_df_imputed=impute_data(imputer,X_df,fmri_key=j, neighbors=neighbors)

        # Varying no. of features
        # for i in tqdm(range(10)):
        for i in tqdm([1]):
            f1_scores=[]
            fig, ax = plt.subplots()
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for k, (train_idx,test_idx) in tqdm(enumerate(cv_outer.split(X_df,y))):
                
                y_train = y[train_idx]
                y_test = y[test_idx]
                
                if model=='NBF':
                    processed_data_path="/home/navid/Dropbox/Repo_2022/Epilep/Epileptogenesis/_data/processed"
                    x_d,x_e,x_fn,x_fp,x_fo,y=load_data(processed_data_path,NBF=True)

                    X_train_dwi,y_train_dwi=drop_nan_index(x_d,y,train_idx)
                    X_train_eeg,y_train_eeg=drop_nan_index(x_e,y,train_idx)
                    X_test_dwi=x_d[test_idx,:]
                    X_test_eeg=x_e[test_idx,:]

                    if j==1:
                        X_train_fmri,y_train_fmri=drop_nan_index(x_fn,y,train_idx)
                        X_test_fmri=x_fn[test_idx,:]
                    elif j==2:
                        X_train_fmri,y_train_fmri=drop_nan_index(x_fp,y,train_idx)
                        X_test_fmri=x_fp[test_idx,:]
                    else:
                        X_train_fmri,y_train_fmri=drop_nan_index(x_fo,y,train_idx)
                        X_test_fmri=x_fo[test_idx,:]

                    # SVM Classifiers
                    fmri_class=nb_svm(X_train_fmri,y_train_fmri)
                    dwi_class=nb_svm(X_train_dwi,y_train_dwi)
                    eeg_class=nb_svm(X_train_eeg,y_train_eeg)

                    fmri_grid = pd.DataFrame(fmri_class.cv_results_)
                    dwi_grid = pd.DataFrame(fmri_class.cv_results_)
                    # eeg_grid = pd.DataFrame(fmri_class.cv_results_)
                    grid_search = fmri_grid.append(dwi_grid)
                    grid_search.to_csv(output_path+'_stat/'+'grid_search_nbf_svm.csv')

                    y_pred,y_prob_true,y_prob_false=naive_bayes_multimodal(fmri_class,X_test_fmri,dwi_class,X_test_dwi,y_test,y_train,eeg_class,X_test_eeg)
                    roc_auc=roc_auc_score(y_test, y_pred)

                else:
                    # Imputing the train and test splits
                    X_train = X_df_imputed[train_idx,:]
                    X_test = X_df_imputed[test_idx,:]
                    # X_train_imputed,X_test_imputed=impute_data(imputer,X_train,X_test,fmri_key=j, neighbors=neighbors)

                    # Fit and test with desired classifier
                    # y_pred,fpr,tpr,roc_auc=model_run(model,i,k,ax,X_train_imputed,X_test_imputed,y_train,y_test,roc_flag,direction='forward')
                    y_pred,fpr,tpr,roc_auc=model_run(model,i,k,ax,X_train,X_test,y_train,y_test,roc_flag,direction,fixed_feat)

                f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
                # if roc_flag:
                #     interp_tpr = np.interp(mean_fpr,fpr,tpr)
                #     interp_tpr[0] = 0.0
                #     tprs.append(interp_tpr)
                aucs.append(roc_auc)

            # Record all results 
            results = results.append({'fMRI':key,'Feats':i+1,'Fixed_feat':fixed_feat,'Imputer':imputer,'Neighbors':neighbors,'AUC: Mean':round(mean(aucs),3),'SEM':round(1.96*stats.sem(aucs,ddof=0),3),'f1: Mean':round(mean(f1_scores),3),'SEM f1':round(1.96*stats.sem(f1_scores,ddof=0),3)},ignore_index=True)

            if ~roc_flag:
                plt.close()

    # if roc_flag:
    #     ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)

    #     mean_tpr = np.mean(tprs, axis=0)
    #     mean_tpr[-1] = 1.0
    #     mean_auc = auc(mean_fpr, mean_tpr)
    #     std_auc = np.std(aucs)
    #     ax.plot(
    #         mean_fpr,
    #         mean_tpr,
    #         color="b",
    #         label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    #         lw=2,
    #         alpha=0.8,
    #     )
    #     # std_tpr = np.std(tprs, axis=0)
    #     # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    #     # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    #     # Using Binomial conf intervals, as laid out in Sourati 2015
    #     [tprs_upper, tprs_lower] = binom_conf_interval(mean_tpr*48, 48, confidence_level=0.95, interval='wilson')  

    #     ax.fill_between(
    #         mean_fpr,
    #         tprs_lower,
    #         tprs_upper,
    #         color="grey",
    #         alpha=0.2,
    #         # label=r"$\pm$ 1 std. dev.",            
    #         label=r'95% level of confidence',
    #     )

    #     ax.set(
    #         xlim=[-0.05, 1.05],
    #         ylim=[-0.05, 1.05],
    #         # title="Receiver operating characteristic example",
    #     )
    #     ax.legend(loc="lower right")
    #     plt.savefig(output_path+'_plot/'+model+'_'+imputer+'_'+str(neighbors)+'_fixed_'+str(fixed_feat)+text+'.png') 
    
    # Export all results to csv
    results.to_csv(output_path+'_stat/'+model+'_'+imputer+'_'+str(neighbors)+'_fixed_'+str(fixed_feat)+text+'.csv')

def model_run(model,i,k,ax,X_train_imputed,X_test_imputed,
                y_train,y_test,roc_flag,direction,fixed_feat):
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

    elif model=='GCCA':
        gcca=None
        gcca=GCCA(latent_dims=i+1)
        # gcca.fit(X_train_imputed[:,0:166],X_train_imputed[:,166:229],X_train_imputed[:,229:232])
        X_train_f,X_train_d,X_train_e=gcca.fit_transform((X_train_imputed[:,0:166],X_train_imputed[:,166:229],X_train_imputed[:,229:232]))
        X_train=np.concatenate((X_train_f,X_train_d), axis=1)
        X_train=np.concatenate((X_train,X_train_e), axis=1)
        X_test_f,X_test_d,X_test_e=gcca.transform((X_test_imputed[:,0:166],X_test_imputed[:,166:229],X_test_imputed[:,229:232]))
        X_test=np.concatenate((X_test_f,X_test_d), axis=1)
        X_test=np.concatenate((X_test,X_test_e), axis=1)

    elif model=='KGCCA':
        kgcca=None
        kgcca=KGCCA(latent_dims=i+1)
        # gcca.fit(X_train_imputed[:,0:166],X_train_imputed[:,166:229],X_train_imputed[:,229:232])
        X_train_f,X_train_d,X_train_e=kgcca.fit_transform((X_train_imputed[:,0:166],X_train_imputed[:,166:229],X_train_imputed[:,229:232]))
        X_train=np.concatenate((X_train_f,X_train_d), axis=1)
        X_train=np.concatenate((X_train,X_train_e), axis=1)
        X_test_f,X_test_d,X_test_e=kgcca.transform((X_test_imputed[:,0:166],X_test_imputed[:,166:229],X_test_imputed[:,229:232]))
        X_test=np.concatenate((X_test_f,X_test_d), axis=1)
        X_test=np.concatenate((X_test,X_test_e), axis=1)

    elif model=='CCA+SFS':
        # Fix feats in one model, sweep feats in other
        cca=None
        cca=CCA(n_components=fixed_feat)
        X_train_f,X_train_d=cca.fit_transform(X_train_imputed[:,0:166],X_train_imputed[:,166:229])
        X_train_cca=np.concatenate((X_train_f,X_train_d), axis=1)
        X_test_f,X_test_d=cca.transform(X_test_imputed[:,0:166],X_test_imputed[:,166:229])
        X_test_cca=np.concatenate((X_test_f,X_test_d), axis=1)
        sfs=None
        sfs=SequentialFeatureSelector(clf, direction=direction, scoring='f1', n_features_to_select=i+1)
        X_train_sfs=sfs.fit_transform(X_train_imputed,y_train)
        X_test_sfs=sfs.transform(X_test_imputed)
        X_train=np.concatenate((X_train_cca,X_train_sfs), axis=1)
        X_test=np.concatenate((X_test_cca,X_test_sfs), axis=1)

    elif model=='CCA+SMIG':
        # Fix feats in one model, sweep feats in other
        cca=None
        cca=CCA(n_components=fixed_feat)
        X_train_f,X_train_d=cca.fit_transform(X_train_imputed[:,0:166],X_train_imputed[:,166:229])
        X_train_cca=np.concatenate((X_train_f,X_train_d), axis=1)
        X_test_f,X_test_d=cca.transform(X_test_imputed[:,0:166],X_test_imputed[:,166:229])
        X_test_cca=np.concatenate((X_test_f,X_test_d), axis=1)
        smig=None
        smig=MMINet(input_dim=232, output_dim=i+1, net='linear')
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
