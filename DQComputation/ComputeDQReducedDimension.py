#feat_iris = [3, 2, 1]
#feat_cancer = [23, 12, 5, 4, 3, 2]
#feat_adult = [8, 4, 2]
#feat_heart = [4, 8, 10]
feat_statlog = [12, 18]
#feat_abalone = [6]
feat_spambase = [43, 52]
feat_bean = [4, 12]
nb_feats_all = [feat_statlog, feat_spambase, feat_bean]

if __name__ == '__main__':
    freeze_support()
    cpus = cpu_count()
    for dataset, nb_feats in zip(['statlog', 'spambase', 'bean'], nb_feats_all):
        df_train = pd.read_csv("dataset/"+dataset+"/trusted_test/"+dataset+"_train.csv")
        df_test = pd.read_csv("dataset/"+dataset+"/trusted_test/"+dataset+"_test.csv")
        feature_importances = pd.read_csv("dataset/"+dataset+"/feature_importances.csv", index_col=0)
        #if dataset == 'bean':
        #    df_train = pd.read_csv("dataset/" + dataset + "/trusted_test/" + dataset + "_og_train.csv")
        #    df_test = pd.read_csv("dataset/" + dataset + "/trusted_test/" + dataset + "_og_test.csv")
        n = df_train.shape[0] + df_test.shape[0]
        y_train = df_train['class'].copy()
        y_test = df_test['class'].copy()
        df_train.drop(columns=['class'], inplace=True)
        df_test.drop(columns=['class'], inplace=True)
        for nb in nb_feats:
            features = feature_importances.nlargest(nb, "FeatureImportance").index
            X_train = df_train.iloc[:, features]
            X_test = df_test.iloc[:, features]
            d = round(nb/n, 4)
            dqm.dq_metric_test_para(X_train, X_test, y_train, y_test, crt_names, models, dataset,
                                    dataset+"_"+str(nb)+"_features_d="+str(d))