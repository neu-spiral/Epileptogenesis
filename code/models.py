from helper import *

def run_estimator(cv_outer,output_path,model,X_df,y,text,options,
                    imputer='KNN',neighbors=3,roc_flag='False',fixed_feat=0,direction='forward'):
    fmri_feat_outer=['Pos']
    # fmri_feat=['Neg','Pos','Ov']
    fmri_feat=['Ov']
    l=1
    j=2
    results=pd.DataFrame()
    roc_data=pd.DataFrame()
    print('Model:',model,', Imputer:',imputer,', Neighbors:',neighbors)

    for key_2 in fmri_feat_outer:
        l+=1
        # if j>3:
        #     break
        print(key_2,'fmri strength outer')

        for key in fmri_feat:
            j+=1
            print(key,'fmri strength inner')

            # Varying no. of features
            # for i in tqdm(range(10)):
            for i in tqdm([2]):
                f1_scores=[]
                sensitivity1 = []
                specificity1 = [] 
                fig, ax = plt.subplots()
                tprs = []
                aucs = []
                mean_fpr = np.linspace(0, 1, 100)

                for k, (train_idx,test_idx) in tqdm(enumerate(cv_outer.split(X_df,y))):
                    
                    y_train = y[train_idx]
                    y_test = y[test_idx]
                    X_train_raw = X_df[train_idx]
                    X_test_raw = X_df[test_idx]
                    
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
                        X_train,X_test=impute_data(imputer,X_train_raw,X_test_raw,fmri_key=j, neighbors=neighbors)
                        X_train_2,X_test_2=impute_data(imputer,X_train_raw,X_test_raw,fmri_key=l, neighbors=neighbors)

                        # Fit and test with desired classifier
                        y_pred,fpr,tpr,roc_auc,X_test_model=model_run(model,i,k,ax,X_train,X_test,X_train_2,X_test_2,y_train,y_test,roc_flag,direction,fixed_feat)   
                        if options=='X_test':
                            if k==0 and options=='X_test':
                                X_model=np.array(X_test_model)
                                y_model=np.array(y_test)
                            else:                          
                                X_test_model=np.array(X_test_model)
                                y_test=np.array(y_test)
                                X_model=np.append(X_model,X_test_model,axis=0)   
                                y_model=np.append(y_model,y_test,axis=0) 

                    
                    cm1 = confusion_matrix(y_test,y_pred)
                    # total1 = sum(sum(cm1))

                    sensitivity1.append(cm1[0,0]/(cm1[0,0]+cm1[0,1]))        
                    specificity1.append(cm1[1,1]/(cm1[1,0]+cm1[1,1]))   
                    f1_scores.append(f1_score(y_test, y_pred, average='weighted'))
                    aucs.append(roc_auc)
                    if roc_flag=='True':
                        interp_tpr = np.interp(mean_fpr,fpr,tpr)
                        interp_tpr[0] = 0.0
                        tprs.append(interp_tpr)
                    
                if options=='X_test':
                    # X_model=np.array(X_test_model)    
                    # y_model=np.array(y_model)  
                    print(X_model.shape)
                    print(y_model.shape)
                    plot_manifold(output_path,model,X_model,y_model,text,options,imputer,neighbors,fixed_feat)               

                # Record all results 
                results = results.append({'fMRI_out':key_2,'fMRI_in':key,'Feats_fix':fixed_feat,'Feats_var':i+1,'Imputer':imputer,'Neighbors':neighbors,'AUC: Mean':round(mean(aucs),3),'SEM':round(1.96*stats.sem(aucs,ddof=0),3),'Sensitivity':round(mean(sensitivity1),3),'Specificity':round(mean(specificity1),3),'f1: Mean':round(mean(f1_scores),3),'SEM f1':round(1.96*stats.sem(f1_scores,ddof=0),3)},ignore_index=True)

                if roc_flag=='False':
                    plt.close()

    if roc_flag=='True':
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

        # Export roc data points to csv
        if options=='roc_data':
            for m in range(mean_fpr.size):
                roc_data = roc_data.append({'mean_fpr':mean_fpr[m],'mean_tpr':mean_tpr[m]},ignore_index=True)
            roc_data.to_csv(output_path+'extra/'+model+'_'+key+'_'+imputer+'_'+str(neighbors)+'_fixed_'+str(fixed_feat)+text+'_roc_pts.csv')

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
        plt.xlabel('1 - Specificity')
        plt.ylabel('Sensitivity')
        plt.savefig(output_path+'_plot/'+model+'_'+key+'_'+imputer+'_'+str(neighbors)+'_fixed_'+str(fixed_feat)+text+'.png') 
    
    # Export all results to csv
    results.to_csv(output_path+'_stat/'+model+'_'+key+'_'+imputer+'_'+str(neighbors)+'_fixed_'+str(fixed_feat)+text+'.csv')

def model_run(model,i,k,ax,X_train_imputed,X_test_imputed,X_train_2,X_test_2,
                y_train,y_test,roc_flag,direction,fixed_feat):
    # SVM Classifier
    clf=None
    clf=make_pipeline(StandardScaler(), SVC(gamma='auto'))

    if model=='SFS':
        sfs=None
        sfs=SequentialFeatureSelector(clf, direction=direction, scoring='roc_auc', n_features_to_select=i+1)
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
        X_train_f,X_train_d=cca.fit_transform(X_train_2[:,0:166],X_train_2[:,166:229])
        X_train_cca=np.concatenate((X_train_f,X_train_d), axis=1)
        X_test_f,X_test_d=cca.transform(X_test_2[:,0:166],X_test_2[:,166:229])
        X_test_cca=np.concatenate((X_test_f,X_test_d), axis=1)
        sfs=None
        sfs=SequentialFeatureSelector(clf, direction=direction, scoring='roc_auc', n_features_to_select=i+1)
        X_train_sfs=sfs.fit_transform(X_train_imputed,y_train)
        X_test_sfs=sfs.transform(X_test_imputed)
        X_train=np.concatenate((X_train_cca,X_train_sfs), axis=1)
        X_test=np.concatenate((X_test_cca,X_test_sfs), axis=1)

    elif model=='CCA+SMIG':
        # Fix feats in one model, sweep feats in other
        cca=None
        cca=CCA(n_components=fixed_feat)
        X_train_f,X_train_d=cca.fit_transform(X_train_2[:,0:166],X_train_2[:,166:229])
        X_train_cca=np.concatenate((X_train_f,X_train_d), axis=1)
        X_test_f,X_test_d=cca.transform(X_test_2[:,0:166],X_test_2[:,166:229])
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

    return y_pred,viz.fpr,viz.tpr,viz.roc_auc,X_test

def plot_manifold(output_path,model,X,y,text,options,
                    imputer='KNN',neighbors=3,fixed_feat=0):

    
    fmri_feat=['Neg','Pos','Ov']
    j=0
    for key in fmri_feat:
        j+=1
        if options=='X_test':
            X_df_imputed=X
        elif options=='no_impute':
            X_df_imputed=X
        else:
            X_df_imputed=impute_data(imputer,X,fmri_key=j, neighbors=neighbors)

        manifold = MDS(n_components=3)
        results = manifold.fit_transform(X_df_imputed)

        # fig, ax = plt.subplots()
        ax = plt.figure(figsize=(8,8)).gca(projection='3d')
        cm = plt.cm.viridis

        # For 3-D
        scat = ax.scatter(
            xs=results[:,0], 
            ys=results[:,1], 
            zs=results[:,2], 
            c=y,
            s=100,
            cmap=cm)

        # For 2-D
        # scat = ax.scatter(
        #     x=results[:,0], 
        #     y=results[:,1], 
        #     c=y,
        #     cmap=cm)
        legend_elem = [Line2D([0], [0], marker='o', color=cm(0.),lw=0,label='No Seizure'),
                        Line2D([0], [0], marker='o', color=cm(1.),lw=0,label='Late Seizure')]

        legend1 = ax.legend(handles=legend_elem,
                            loc="upper right")

        ax.add_artist(legend1)
        ax.set_xlabel('Component-1')
        ax.set_ylabel('Component-2')
        ax.set_zlabel('Component-3')

        plt.tight_layout()     
        plt.savefig(output_path+'_plot/'+model+'_'+key+'_'+imputer+'_'+str(neighbors)+'_fixed_'+str(fixed_feat)+text+'_manifold.png')     