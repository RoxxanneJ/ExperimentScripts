import pandas as pd
import utils.data_quality_metric as dqm
from multiprocessing import freeze_support

# We initialize the number of features selected

feat_iris = [3, 2, 1]
feat_cancer = [23, 12, 5, 4, 3, 2]
feat_adult = [8, 4, 2]
feat_heart = [12, 10, 8, 6, 4, 3, 2, 1]
feat_statlog = [20, 18, 12, 10, 8, 6, 4, 3, 2, 1]
feat_abalone = [6, 4, 3, 2, 1]
feat_spambase = [52, 43, 37, 28, 19, 10, 6, 4, 3, 2, 1]
feat_bean = [12, 11, 8, 6, 4, 3, 2, 1]
nb_feats_all = [feat_iris, feat_cancer, feat_adult, feat_heart, feat_abalone, feat_spambase, feat_statlog, feat_bean]

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
datasets = ["iris", "cancer", "adult", "heart", "abalone", "spambase", "statlog", "bean"]
crt_names = ["missing", "outlier", "fuzzing"]

if __name__ == '__main__':
    freeze_support()
    for dataset, nb_feats in zip(datasets, nb_feats_all):
        df_train = pd.read_csv("../DataDeterioration/DeterioratedDatasets/" + dataset + "/trusted_test/" + dataset +
                               "_train.csv")
        df_test = pd.read_csv("../DataDeterioration/DeterioratedDatasets/" + dataset + "/trusted_test/" + dataset +
                              "_test.csv")

        # This file contains the features and their orders of importance
        feature_importances = pd.read_csv("../DataDeterioration/DeterioratedDatasets/" + dataset +
                                          "/feature_importances.csv", index_col=0)

        n = df_train.shape[0] + df_test.shape[0]
        y_train = df_train['class'].copy()
        y_test = df_test['class'].copy()
        df_train.drop(columns=['class'], inplace=True)
        df_test.drop(columns=['class'], inplace=True)
        for nb in nb_feats:
            # We only keep the nb_feats most important features
            features = feature_importances.nlargest(nb, "FeatureImportance").index
            X_train = df_train.iloc[:, features]
            X_test = df_test.iloc[:, features]
            d = round(nb/n, 4)
            dqm.dq_metric_test_para(X_train, X_test, y_train, y_test, crt_names, models, dataset,
                                    dataset + "_" + str(nb) + "_features_d=" + str(d))
