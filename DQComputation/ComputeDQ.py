import pandas as pd
import utils.data_quality_metric as dqm
from multiprocessing import freeze_support

models = ['logistic regression', 'knn', 'decision tree', 'random forest', 'ada boost', 'naive bayes', 'xgboost',
          'svc', 'gaussian process', 'mlp', 'sgd', 'gradient boosting']
datasets = ["iris", "cancer", "adult", "heart", "abalone", "spambase", "statlog", "bean"]
errors = ["missing", "outlier", "fuzzing", "missing_outlier"]
crt_names = ["missing", "outlier", "fuzzing"]

if __name__ == '__main__':
    freeze_support()
    for error in errors:
        for dataset in datasets:
            path = "../DataDeterioration/DeterioratedDatasets/" + dataset + "/" + error + "/"
            for p in range(5, 55, 5):
                name = dataset + "_" + error + "_" + str(p)
                df = pd.read_csv(path + name + ".csv")
                df.dropna(inplace=True)
                dqm.dq_metric_para(30, df, crt_names, models, dataset, name)

