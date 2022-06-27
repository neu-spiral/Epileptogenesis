from helper import *

def load_data(processed_data_path):
    fmri_features=pd.read_csv(f"{processed_data_path}/fMRI/fMRI_features_AAL.csv",index_col=0)

    # print('fMRI Subject IDs')
    # print(fmri_features["Subject"])
    dwi_features = pd.read_csv(f"{processed_data_path}/DWI/subs_jan_2022.csv")

    dwi_features["Subject"]=dwi_features["ID"].str[:9]
    dwi_features["Late Seizure Label"]=dwi_features["Label"]
    dwi_features=dwi_features.drop("Label",axis=1)

    eeg_features=pd.read_csv(f"{processed_data_path}/EEG/EEG_features_v0.csv",index_col=0)

    # need to load EEG and DWI features and sort out which subjects to use programatically

    all_features_df=fmri_features.set_index("Subject").join(dwi_features.set_index("Subject"),how="outer",lsuffix=" fMRI",rsuffix=" DWI").reset_index()
    all_features_df=all_features_df.set_index("Subject").join(eeg_features.set_index("Subject"),how="outer",lsuffix=" Mix",rsuffix=" EEG").reset_index()
    all_features_df["Late Seizure Label EEG"]=all_features_df["Late Seizure Label"]


    all_features_df["Late Seizure Label"]=(all_features_df["Late Seizure Label fMRI"].fillna(0)+all_features_df["Late Seizure Label DWI"].fillna(0)+all_features_df["Late Seizure Label EEG"].fillna(0))>0

    # make np array features for classification

    dwi_columns=remove_non_features([*dwi_features])
    eeg_columns=remove_non_features([*eeg_features])
    fmri_columns=remove_non_features([*fmri_features])

    #dwi 
    y = all_features_df["Late Seizure Label"].to_numpy()
    X_dwi = all_features_df[dwi_columns].to_numpy()

    # fMRI

    overlap_columns=[]
    mean_str_pos_columns=[]
    mean_str_neg_columns=[]

    for col in fmri_columns:
        if "Overlap AAL" in col:
            overlap_columns.append(col)
        elif "Pos AAL" in col:
            mean_str_pos_columns.append(col)
        elif "Neg AAL" in col:
            mean_str_neg_columns.append(col)

    X_over_aal=all_features_df[overlap_columns].to_numpy()
    X_pos_str_aal=all_features_df[mean_str_pos_columns].to_numpy()

    all_features_df[mean_str_neg_columns]=-1*all_features_df[mean_str_neg_columns]
    X_neg_str_aal=all_features_df[mean_str_neg_columns].to_numpy()
    #eeg 
    X_eeg=all_features_df[eeg_columns].to_numpy()

    #all_features=np.concatenate([X_over_aal,X_dwi,X_eeg],axis=1)
    fmri_len=X_over_aal.shape[1]
    dwi_len=X_dwi.shape[1]
    eeg_len=X_eeg.shape[1]

    fmri_ind=[*range(0,fmri_len)]
    dwi_ind=[*range(fmri_len,fmri_len+dwi_len)]
    eeg_ind=[*range(fmri_len+dwi_len,fmri_len+dwi_len+eeg_len)]

    select_fmri_ov=ColumnTransformer([("fMRI ov",'passthrough',overlap_columns)])
    select_fmri_pos=ColumnTransformer([("fMRI pos",'passthrough',mean_str_pos_columns)])
    select_fmri_neg=ColumnTransformer([("fMRI neg",'passthrough',mean_str_neg_columns)])
    select_fmri=ColumnTransformer([("fMRI",'passthrough',[*mean_str_pos_columns,*mean_str_neg_columns,*overlap_columns])])

    select_eeg=ColumnTransformer([("EEG",'passthrough',eeg_columns)])
    select_dwi=ColumnTransformer([("DWI",'passthrough',dwi_columns)])

    select_all_pos=ColumnTransformer([("ALL",'passthrough',[*mean_str_pos_columns,*eeg_columns,*dwi_columns])])
    select_all_neg=ColumnTransformer([("ALL",'passthrough',[*mean_str_neg_columns,*eeg_columns,*dwi_columns])])
    select_all_ov=ColumnTransformer([("ALL",'passthrough',[*overlap_columns,*eeg_columns,*dwi_columns])])

    # Fusion excluding EEG
    dwi_fmri_pos=ColumnTransformer([("ALL",'passthrough',[*mean_str_pos_columns,*dwi_columns])])
    dwi_fmri_neg=ColumnTransformer([("ALL",'passthrough',[*mean_str_neg_columns,*dwi_columns])])
    dwi_fmri_ov=ColumnTransformer([("ALL",'passthrough',[*overlap_columns,*dwi_columns])])

    select_all=ColumnTransformer([("ALL",'passthrough',[*mean_str_pos_columns,*mean_str_neg_columns,*overlap_columns,*eeg_columns,*dwi_columns])])
    class_features_df=all_features_df
    class_features_df=class_features_df.drop(["ID","Subject","Subject Number"],axis=1)

    return class_features_df,y

def impute_data(imputer,train_df,test_df,fmri_key,neighbors=3):
    if imputer=="KNN":
        imputer_knn=KNNImputer(n_neighbors=neighbors)

        # Select appropriate columns for chosen fmri strength
        x_fmri=train_df[:,(fmri_key-1)*166:fmri_key*166]
        x_dmri=train_df[:,499:562]
        x_eeg=train_df[:,564:567]
        train_df_f=np.append(x_fmri,x_dmri,axis=1)
        train_df_f=np.append(train_df_f,x_eeg,axis=1)

        x_fmri=test_df[:,(fmri_key-1)*166:fmri_key*166]
        x_dmri=test_df[:,499:562]
        x_eeg=test_df[:,564:567]
        test_df_f=np.append(x_fmri,x_dmri,axis=1)
        test_df_f=np.append(test_df_f,x_eeg,axis=1)
        
        imputed_train=imputer_knn.fit_transform(train_df_f)
        imputed_test=imputer_knn.transform(test_df_f)

        return imputed_train,imputed_test
